# dark mode
import streamlit as st
import yfinance as yf
import pandas as pd

st.write("""
# App Preço de Ações
O gráfico abaixo representa a evolução do preço das ações brasileiras ao longo dos anos
""")

@st.cache_data
def carregar_dados(empresas):
    dados_acao = yf.Tickers(empresas)
    precos_acao = dados_acao.history(period='1d', start='2015-01-01', end='2024-07-01')
    precos_acao = precos_acao["Close"]
    return precos_acao

dados  = carregar_dados("ITUB4.SA BBAS3.SA VALE3.SA ABEV3.SA MGLU3.SA PETR4.SA GGBR4.SA")

lista_acoes = st.multiselect("Escolha as ações para exibir no gráfico", dados.columns)
print(lista_acoes)
dados = dados[lista_acoes]
grafico = st.line_chart(dados)
