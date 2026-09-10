# app resultado de ações
import streamlit as st
import yfinance as yf
import pandas as pd

st.write("""
# App Preço de Ações
O gráfico abaixo representa a evolução do preço das ações do Itaú (ITUB4) ao longo dos anos
""")

def carregar_dados(empresa):
    dados_acao = yf.Ticker(empresa)
    precos_acao = dados_acao.history(period='1d', start='2000-01-01', end='2024-07-01')
    # precos_acao = pd.read_csv("BaseItau.csv")
    # precos_acao["Date"] = pd.to_datetime(precos_acao["Date"])
    # precos_acao = precos_acao.set_index("Date")
    precos_acao = precos_acao[["Close"]]
    return precos_acao

dados  = carregar_dados("ITUB4.SA")
print(dados)
grafico = st.line_chart(dados)

st.write("""
# Fim do app
""")

