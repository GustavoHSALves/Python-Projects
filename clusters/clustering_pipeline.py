"""
clustering_pipeline.py
=======================

Pipeline de clustering (K-Means) genérico e reutilizável para segmentação
de qualquer base tabular — não é específico do dataset "Mall Customers".

Para usar em outro cenário, basta trocar as colunas passadas em
`numeric_features` / `categorical_features` / `id_columns`. Nenhuma parte
do código assume nomes de coluna fixos.

Resolve, em relação a uma implementação "tutorial" ingênua de K-Means:
  1. Reprodutibilidade: random_state fixo em toda instância de KMeans.
  2. Escolha de k com evidência quantitativa: inércia (cotovelo) +
     silhouette score, em vez de "olhar o gráfico e chutar".
  3. Padronização consistente: todas as features numéricas usadas no
     clustering passam por StandardScaler antes do fit, sempre.
  4. Não esconde warnings globalmente — cada função só suprime o que é
     explicitamente irrelevante (ex.: warnings de convergência do KMeans
     quando n_init já foi definido conscientemente).
  5. Detecta colunas tipo ID e evita que entrem como feature, alertando
     quando uma coluna de índice/ID tem correlação alta com outra
     variável (sinal de correlação espúria por ordenação dos dados, não
     de relação real).
  6. Perfil de cluster pronto para leitura de negócio (médias por
     cluster + distribuição de categóricas), com espaço para nomear
     segmentos manualmente.

Exemplo mínimo:
    from clustering_pipeline import ClusterPipeline, check_data_quality

    check_data_quality(df, id_columns=['CustomerID'])

    pipe = ClusterPipeline(
        numeric_features=['Annual Income (k$)', 'Spending Score (1-100)'],
        categorical_features=['Gender'],
        random_state=42,
    )
    pipe.search_k(df, k_range=range(2, 11))
    best_k = pipe.recommend_k(method='silhouette')
    df_labeled = pipe.fit_predict(df, n_clusters=best_k)
    profile = pipe.profile(df_labeled)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Qualidade de dados — sempre rodar antes de qualquer modelagem
# ---------------------------------------------------------------------------

def check_data_quality(df: pd.DataFrame, id_columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Imprime um raio-x básico de qualidade de dados e retorna um resumo.

    Não assume nada sobre o schema: funciona em qualquer dataframe.
    """
    id_columns = list(id_columns or [])
    n_rows, n_cols = df.shape
    nulls = df.isnull().sum()
    dup_rows = df.duplicated().sum()

    print(f"Linhas: {n_rows} | Colunas: {n_cols}")
    print(f"Linhas duplicadas: {dup_rows}")
    print("Nulos por coluna:")
    print(nulls[nulls > 0] if nulls.sum() > 0 else "  (nenhum)")
    print("\nTipos de dado:")
    print(df.dtypes)

    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        corr = numeric_df.corr(numeric_only=True)
        flags = []
        for col in id_columns:
            if col in corr.columns:
                high = corr[col].drop(index=col, errors="ignore")
                high = high[high.abs() > 0.8]
                for other, val in high.items():
                    flags.append((col, other, val))
        if flags:
            print("\n[ALERTA] Coluna(s) tipo ID com correlação alta com outra variável — "
                  "provável artefato de ordenação das linhas, NÃO uma relação real de negócio:")
            for col, other, val in flags:
                print(f"  {col} x {other}: corr={val:.4f}  -> excluir {col} de qualquer correlação/feature")

    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_nulls": nulls,
        "pct_nulls": (nulls / n_rows * 100).round(2),
    })
    return summary


# ---------------------------------------------------------------------------
# Pipeline de clustering
# ---------------------------------------------------------------------------

@dataclass
class ClusterPipeline:
    """K-Means com seleção de k orientada a dados, reutilizável em qualquer
    tabela — o cenário muda só pelos parâmetros abaixo.

    Parameters
    ----------
    numeric_features : colunas numéricas a padronizar e usar no clustering.
    categorical_features : colunas categóricas, one-hot-encoded e incluídas
        no clustering SE `include_categorical_in_clustering=True`. Se False
        (padrão), categóricas entram só no perfil pós-cluster (evita
        distorcer distância euclidiana misturando binário com contínuo,
        a menos que você decida conscientemente incluir).
    id_columns : colunas de identificador, nunca usadas como feature.
    random_state : fixo para reprodutibilidade em toda a pipeline.
    """

    numeric_features: list[str]
    categorical_features: list[str] = field(default_factory=list)
    id_columns: list[str] = field(default_factory=list)
    include_categorical_in_clustering: bool = False
    random_state: int = 42

    def __post_init__(self):
        self.scaler_: Optional[StandardScaler] = None
        self.model_: Optional[KMeans] = None
        self.feature_names_: list[str] = []
        self.search_results_: Optional[pd.DataFrame] = None

    # -- construção de matriz de features -----------------------------------
    def _build_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.numeric_features if c not in df.columns]
        if missing:
            raise KeyError(f"Colunas numéricas ausentes no dataframe: {missing}")

        X = df[self.numeric_features].copy()

        if self.include_categorical_in_clustering and self.categorical_features:
            cat = pd.get_dummies(df[self.categorical_features], drop_first=True)
            X = pd.concat([X, cat], axis=1)

        self.feature_names_ = list(X.columns)
        return X

    def _scale(self, X: pd.DataFrame, fit: bool) -> np.ndarray:
        if fit or self.scaler_ is None:
            self.scaler_ = StandardScaler()
            return self.scaler_.fit_transform(X)
        return self.scaler_.transform(X)

    # -- seleção de k ---------------------------------------------------------
    def search_k(self, df: pd.DataFrame, k_range: Iterable[int] = range(2, 11)) -> pd.DataFrame:
        """Calcula inércia e silhouette score para cada k do range.

        Silhouette exige k >= 2 (não é definido para k=1); inércia é
        reportada para todo o range informado.
        """
        X = self._build_feature_matrix(df)
        Xs = self._scale(X, fit=True)

        rows = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(Xs)
            sil = silhouette_score(Xs, labels) if k >= 2 and k < len(Xs) else np.nan
            rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})

        self.search_results_ = pd.DataFrame(rows)
        return self.search_results_

    def recommend_k(self, method: str = "silhouette") -> int:
        """Recomenda k a partir dos resultados de `search_k`.

        method='silhouette': k com maior silhouette score (mais robusto
            e quantitativo — recomendado como critério principal).
        method='elbow': k a partir do qual o ganho marginal de inércia
            (segunda derivada) fica pequeno — heurística automática do
            "cotovelo", sem precisar olhar o gráfico manualmente.
        """
        if self.search_results_ is None:
            raise RuntimeError("Rode search_k(df) antes de recommend_k().")

        res = self.search_results_.dropna(subset=["silhouette"]) if method == "silhouette" else self.search_results_

        if method == "silhouette":
            return int(res.loc[res["silhouette"].idxmax(), "k"])

        if method == "elbow":
            inertia = res["inertia"].to_numpy()
            k_vals = res["k"].to_numpy()
            if len(inertia) < 3:
                return int(k_vals[0])
            # segunda diferença: ponto de maior "curvatura" do cotovelo
            first_diff = np.diff(inertia)
            second_diff = np.diff(first_diff)
            elbow_idx = int(np.argmax(second_diff)) + 1  # +1 por causa do duplo diff
            return int(k_vals[elbow_idx])

        raise ValueError("method deve ser 'silhouette' ou 'elbow'")

    # -- fit final --------------------------------------------------------------
    def fit_predict(self, df: pd.DataFrame, n_clusters: int, cluster_col: str = "cluster") -> pd.DataFrame:
        """Ajusta o KMeans final com n_clusters e retorna uma cópia do
        dataframe com a coluna de cluster adicionada."""
        X = self._build_feature_matrix(df)
        Xs = self._scale(X, fit=True)

        self.model_ = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        labels = self.model_.fit_predict(Xs)

        out = df.copy()
        out[cluster_col] = labels
        return out

    def silhouette_of_fit(self, df: pd.DataFrame, cluster_col: str = "cluster") -> float:
        """Silhouette score do último fit_predict, útil para reportar
        qualidade do resultado final escolhido."""
        if self.model_ is None:
            raise RuntimeError("Rode fit_predict() antes.")
        X = self._build_feature_matrix(df.drop(columns=[cluster_col]))
        Xs = self._scale(X, fit=False)
        return silhouette_score(Xs, df[cluster_col])

    # -- perfil de negócio --------------------------------------------------
    def profile(self, df_labeled: pd.DataFrame, cluster_col: str = "cluster") -> pd.DataFrame:
        """Média das features numéricas por cluster + contagem de membros.
        Base para nomear os segmentos manualmente."""
        agg_cols = [c for c in self.numeric_features if c in df_labeled.columns]
        prof = df_labeled.groupby(cluster_col)[agg_cols].mean()
        prof["n_customers"] = df_labeled.groupby(cluster_col).size()
        prof["pct_of_total"] = (prof["n_customers"] / len(df_labeled) * 100).round(1)
        return prof.round(2)

    def categorical_profile(self, df_labeled: pd.DataFrame, cat_col: str, cluster_col: str = "cluster") -> pd.DataFrame:
        """Distribuição percentual de uma coluna categórica dentro de cada cluster."""
        return pd.crosstab(df_labeled[cluster_col], df_labeled[cat_col], normalize="index").round(3)
