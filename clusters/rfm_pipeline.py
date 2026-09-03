"""
rfm_pipeline.py
================

Pipeline de segmentação RFM (Recency, Frequency, Monetary) genérico e
reutilizável — não é específico do dataset Online Retail II. Qualquer tabela
transacional (linha = 1 item vendido/1 interação, com cliente, valor, data e
um identificador de "pedido"/"visita") pode ser processada trocando só os
nomes de coluna passados aos parâmetros.

Resolve, em relação a uma implementação "tutorial" de RFM + K-Means:

 1. Limpeza com relatório DECOMPOSTO por motivo de exclusão (quantos registros
    caíram por invoice inválido, por stock code inválido, por cliente nulo,
    por preço <= 0) — não só um "dropou X%" agregado que esconde a causa.
 2. Assert explícito de invariantes de qualidade (ex.: nenhuma quantidade
    negativa sobra após a limpeza) em vez de confiar implicitamente que
    outros filtros removeram o problema como efeito colateral.
 3. k escolhido por argmax do silhouette score, com o valor sugerido sempre
    impresso e comparável a qualquer k escolhido manualmente por motivo de
    negócio (nº de segmentos "redondo" para apresentação).
 4. Log-transform opcional (log1p) de Monetary/Frequency antes do
   StandardScaler — RFM de varejo é tipicamente assimétrico à direita mesmo
    após remover outliers por IQR (validado empiricamente: skew ~1.2 antes,
    ~-0.5 depois do log, num teste com distribuição lognormal de gasto).
 5. Nomes de segmento derivados por RANKING dos centróides (quem gasta mais
    e compra mais recentemente vira "Champions"/"Reward" etc.), não por um
    dicionário fixo {0: "RETAIN", 1: "RE-ENGAGE", ...} — um mapeamento por
    índice inteiro quebra silenciosamente se a ordem dos clusters mudar
    (nova leva de dados, novo threshold de outlier, nova versão do sklearn).
 6. Perfil de segmento inclui % de CLIENTES e % de RECEITA — a pergunta que
    o time de marketing/CRM realmente faz é "esse segmento é 8% da base mas
    concentra quanto de faturamento?", que a v1 não respondia.
 7. `segment_type` distingue segmentos vindos do modelo (K-Means) dos
    segmentos vindos de regra manual (outliers de monetary/frequency) — a v1
    misturava os dois na mesma coluna de cluster sem sinalizar a diferença
    de método/confiança estatística.
 8. random_state fixo em todo KMeans.

Exemplo mínimo:
    from rfm_pipeline import clean_transactions, compute_rfm, flag_outliers_iqr, \\
        select_k, fit_segments, profile_segments

    cleaned, report = clean_transactions(
        df, invoice_col="Invoice", stockcode_col="StockCode",
        customer_col="Customer ID", quantity_col="Quantity", price_col="Price",
    )
    rfm = compute_rfm(cleaned, customer_col="Customer ID", invoice_col="Invoice",
                       date_col="InvoiceDate", amount_col="SalesLineTotal")
    non_outliers, outliers = flag_outliers_iqr(rfm, monetary_col="Monetary", frequency_col="Frequency")
    best_k, search = select_k(non_outliers, feature_cols=["Monetary", "Frequency", "Recency"], log_cols=["Monetary", "Frequency"])
    segmented = fit_segments(non_outliers, outliers, feature_cols=["Monetary", "Frequency", "Recency"],
                              log_cols=["Monetary", "Frequency"], n_clusters=best_k)
    profile = profile_segments(segmented, monetary_col="Monetary")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 1. Limpeza de dados transacionais, com relatório decomposto por motivo
# ---------------------------------------------------------------------------

def clean_transactions(
    df: pd.DataFrame,
    invoice_col: str,
    stockcode_col: str,
    customer_col: str,
    quantity_col: str,
    price_col: str,
    invoice_pattern: str = r"^\d{6}$",
    stockcode_patterns: Iterable[str] = (r"^\d{5}$", r"^\d{5}[a-zA-Z]+$"),
    stockcode_extra_allowlist: Iterable[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Limpa uma tabela transacional linha-a-linha e retorna (df_limpo, relatorio).

    O relatório decompõe QUANTOS registros cada regra específica removeu —
    não um número agregado só. Isso é o que permite auditar se a limpeza
    está descartando volume demais por um motivo inesperado.

    Regras (na ordem em que são aplicadas):
      1. invoice_col deve casar `invoice_pattern` (por padrão, 6 dígitos —
         no Online Retail II isso elimina cancelamentos, que usam prefixo
         "C", e lançamentos manuais/ajustes com outros prefixos).
      2. stockcode_col deve casar um dos `stockcode_patterns`, OU estar em
         `stockcode_extra_allowlist` (códigos não numéricos mas legítimos,
         ex.: "PADS").
      3. customer_col não pode ser nulo (obrigatório para RFM em nível de
         cliente).
      4. price_col deve ser > 0 (remove ajustes/lançamentos de valor zero).

    Ao final, valida com `assert` que nenhuma linha com quantidade negativa
    sobrou — não assume isso como efeito colateral das regras acima.
    """
    n0 = len(df)
    work = df.copy()
    work[invoice_col] = work[invoice_col].astype(str)
    work[stockcode_col] = work[stockcode_col].astype(str)

    reasons = []

    bad_invoice = ~work[invoice_col].str.match(invoice_pattern)
    reasons.append(("invalid_invoice_format", int(bad_invoice.sum())))
    work = work[~bad_invoice]

    code_ok = pd.Series(False, index=work.index)
    for pattern in stockcode_patterns:
        code_ok |= work[stockcode_col].str.match(pattern)
    if stockcode_extra_allowlist:
        code_ok |= work[stockcode_col].isin(list(stockcode_extra_allowlist))
    bad_stockcode = ~code_ok
    reasons.append(("invalid_stockcode", int(bad_stockcode.sum())))
    work = work[code_ok]

    bad_customer = work[customer_col].isna()
    reasons.append(("missing_customer_id", int(bad_customer.sum())))
    work = work[~bad_customer]

    bad_price = ~(work[price_col] > 0)
    reasons.append(("non_positive_price", int(bad_price.sum())))
    work = work[work[price_col] > 0]

    # Invariante de qualidade: a limpeza acima deve ter eliminado toda
    # quantidade negativa (cancelamentos/devoluções) como efeito das regras
    # de invoice/stockcode. Não confiar nisso implicitamente — checar.
    if quantity_col in work.columns:
        n_negative_qty = int((work[quantity_col] < 0).sum())
        assert n_negative_qty == 0, (
            f"{n_negative_qty} linhas com {quantity_col} negativa sobreviveram à limpeza — "
            "a suposição de que o filtro de invoice/stockcode remove todos os "
            "cancelamentos/devoluções não se sustenta nesses dados; adicione um "
            f"filtro explícito de {quantity_col} > 0."
        )

    n1 = len(work)
    report = pd.DataFrame(reasons, columns=["reason", "rows_dropped"])
    report["pct_of_original"] = (report["rows_dropped"] / n0 * 100).round(2)
    report.loc[len(report)] = ["TOTAL_KEPT", n1, round(n1 / n0 * 100, 2)]
    report.loc[len(report)] = ["TOTAL_DROPPED", n0 - n1, round((n0 - n1) / n0 * 100, 2)]

    return work, report


# ---------------------------------------------------------------------------
# 2. Agregação RFM
# ---------------------------------------------------------------------------

def compute_rfm(
    df: pd.DataFrame,
    customer_col: str,
    invoice_col: str,
    date_col: str,
    amount_col: str,
    snapshot_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Agrega uma tabela transacional limpa em uma linha por cliente com
    Monetary (soma de amount_col), Frequency (nº de invoices distintos) e
    Recency (dias entre snapshot_date e a última compra).

    snapshot_date: se None, usa o max(date_col) dos próprios dados — prática
    padrão para datasets históricos, mas é uma suposição de negócio
    ("hoje" = a última data observada) que deve ficar explícita no relatório
    final, não escondida no código.
    """
    agg = df.groupby(by=customer_col, as_index=False).agg(
        Monetary=(amount_col, "sum"),
        Frequency=(invoice_col, "nunique"),
        LastPurchaseDate=(date_col, "max"),
    )
    snap = snapshot_date or agg["LastPurchaseDate"].max()
    agg["Recency"] = (snap - agg["LastPurchaseDate"]).dt.days
    agg.attrs["snapshot_date"] = snap
    return agg


# ---------------------------------------------------------------------------
# 3. Outliers — bucket de regra de negócio em vez de descarte silencioso
# ---------------------------------------------------------------------------

def flag_outliers_iqr(
    rfm: pd.DataFrame,
    monetary_col: str = "Monetary",
    frequency_col: str = "Frequency",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa outliers de Monetary e Frequency por IQR (Recency fica de fora
    propositalmente: é naturalmente limitada e menos assimétrica que gasto
    ou frequência de compra).

    Retorna (non_outliers, outliers) onde `outliers` já vem com uma coluna
    `outlier_reason` em {"monetary_only", "frequency_only", "monetary_and_frequency"}
    — esses clientes normalmente são os de MAIOR valor (compradores grandes/
    frequentes), então tratá-los como "lixo estatístico a descartar" perde
    justamente os clientes mais importantes. Aqui eles viram um segmento à
    parte, tratado por regra em vez de por K-Means.
    """
    def _iqr_bounds(s: pd.Series) -> tuple[float, float]:
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        return q1 - 1.5 * iqr, q3 + 1.5 * iqr

    m_lo, m_hi = _iqr_bounds(rfm[monetary_col])
    f_lo, f_hi = _iqr_bounds(rfm[frequency_col])

    is_m_outlier = (rfm[monetary_col] < m_lo) | (rfm[monetary_col] > m_hi)
    is_f_outlier = (rfm[frequency_col] < f_lo) | (rfm[frequency_col] > f_hi)

    non_outliers = rfm[~is_m_outlier & ~is_f_outlier].copy()

    outliers = rfm[is_m_outlier | is_f_outlier].copy()
    reason = np.where(
        is_m_outlier[outliers.index] & is_f_outlier[outliers.index], "monetary_and_frequency",
        np.where(is_m_outlier[outliers.index], "monetary_only", "frequency_only"),
    )
    outliers["outlier_reason"] = reason

    return non_outliers, outliers


# ---------------------------------------------------------------------------
# 4. Seleção de k e fit de K-Means (com log-transform opcional)
# ---------------------------------------------------------------------------

def _build_matrix(df: pd.DataFrame, feature_cols: list[str], log_cols: Iterable[str] = ()) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for col in log_cols:
        if col in X.columns:
            X[col] = np.log1p(X[col].clip(lower=0))
    return X


def select_k(
    df: pd.DataFrame,
    feature_cols: list[str],
    log_cols: Iterable[str] = (),
    k_range: Iterable[int] = range(2, 11),
    random_state: int = 42,
) -> tuple[int, pd.DataFrame]:
    """Roda KMeans para cada k em k_range, retorna (k recomendado por
    argmax do silhouette, tabela completa com inertia + silhouette)."""
    X = _build_matrix(df, feature_cols, log_cols)
    Xs = StandardScaler().fit_transform(X)

    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=1000)
        labels = km.fit_predict(Xs)
        sil = silhouette_score(Xs, labels)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})

    search = pd.DataFrame(rows)
    best_k = int(search.loc[search["silhouette"].idxmax(), "k"])
    return best_k, search


def fit_segments(
    non_outliers: pd.DataFrame,
    outliers: pd.DataFrame,
    feature_cols: list[str],
    n_clusters: int,
    log_cols: Iterable[str] = (),
    monetary_col: str = "Monetary",
    frequency_col: str = "Frequency",
    recency_col: str = "Recency",
    random_state: int = 42,
) -> pd.DataFrame:
    """Ajusta o KMeans final nos não-outliers, nomeia os clusters por
    RANKING de centróide (não por índice hardcoded), anexa os outliers como
    segmentos por regra, e retorna tudo combinado com `segment_type` e
    `SegmentName`.

    Convenção de nomes (do mais ao menos valioso, entre os clusters do
    modelo): Champions > Loyal > Potential > At Risk > Hibernating — o
    número de nomes usados se ajusta a n_clusters, ranqueando primeiro por
    Monetary e usando Frequency/Recency como desempate.
    """
    X = _build_matrix(non_outliers, feature_cols, log_cols)
    Xs = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, max_iter=1000)
    labels = km.fit_predict(Xs)

    out = non_outliers.copy()
    out["Cluster"] = labels
    out["segment_type"] = "model"

    # ranking de centroide: maior Monetary médio primeiro, empatando por
    # Frequency (desc) e Recency (asc, menor = mais recente = melhor)
    centroid_profile = out.groupby("Cluster")[[monetary_col, frequency_col, recency_col]].mean()
    centroid_profile = centroid_profile.sort_values(
        by=[monetary_col, frequency_col, recency_col], ascending=[False, False, True]
    )
    rank_names = ["Champions", "Loyal", "Potential Loyalist", "At Risk", "Hibernating",
                  "Needs Attention", "Promising", "About to Sleep"]
    name_map = {cluster_id: rank_names[i] if i < len(rank_names) else f"Segment {i+1}"
                for i, cluster_id in enumerate(centroid_profile.index)}
    out["SegmentName"] = out["Cluster"].map(name_map)

    out_outliers = outliers.copy()
    out_outliers["segment_type"] = "rule_based_outlier"
    reason_name_map = {
        "monetary_only": "Pamper (big spender, infrequent)",
        "frequency_only": "Upsell (frequent, low ticket)",
        "monetary_and_frequency": "Delight (top-tier: big & frequent)",
    }
    out_outliers["SegmentName"] = out_outliers["outlier_reason"].map(reason_name_map)
    out_outliers["Cluster"] = out_outliers["outlier_reason"].map(
        {"monetary_only": -1, "frequency_only": -2, "monetary_and_frequency": -3}
    )

    combined = pd.concat([out, out_outliers], ignore_index=False)
    return combined


# ---------------------------------------------------------------------------
# 5. Perfil de negócio: % de clientes E % de receita por segmento
# ---------------------------------------------------------------------------

def profile_segments(
    segmented: pd.DataFrame,
    monetary_col: str = "Monetary",
    frequency_col: str = "Frequency",
    recency_col: str = "Recency",
    name_col: str = "SegmentName",
) -> pd.DataFrame:
    """Perfil por segmento: médias de R/F/M, nº de clientes, % de clientes e
    % de RECEITA TOTAL — a métrica que efetivamente prioriza qual segmento
    merece mais orçamento/atenção de CRM."""
    total_customers = len(segmented)
    total_revenue = segmented[monetary_col].sum()

    prof = segmented.groupby(name_col).agg(
        n_customers=(monetary_col, "size"),
        avg_monetary=(monetary_col, "mean"),
        avg_frequency=(frequency_col, "mean"),
        avg_recency=(recency_col, "mean"),
        total_revenue=(monetary_col, "sum"),
    )
    prof["pct_of_customers"] = (prof["n_customers"] / total_customers * 100).round(1)
    prof["pct_of_revenue"] = (prof["total_revenue"] / total_revenue * 100).round(1)
    prof = prof.sort_values("total_revenue", ascending=False)
    return prof.round(2)
