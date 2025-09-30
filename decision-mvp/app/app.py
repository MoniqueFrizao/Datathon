# app/app.py
import streamlit as st
from model_utils import carregar_modelo, fazer_inferencia
from preprocessing import preprocessar_dados
from feature_engineering import transformar_features

st.set_page_config(page_title="Decision Match IA", layout="centered")
st.title("🔍 IA para Recrutamento - Decision")

st.markdown("Preencha os dados do candidato para avaliar a compatibilidade com a vaga.")

# Inputs simulados
nome = st.text_input("Nome do candidato")
idade = st.number_input("Idade", min_value=18, max_value=99)
experiencia = st.selectbox("Experiência em anos", [0, 1, 2, 3, 5, 10])
linguagem = st.multiselect("Linguagens que domina", ["Python", "Java", "C#", "JavaScript", "SQL"])
motivacao = st.slider("Motivação para a vaga (0 a 10)", 0, 10)

if st.button("Avaliar"):
    dados = {
        "nome": nome,
        "idade": idade,
        "experiencia": experiencia,
        "linguagem": linguagem,
        "motivacao": motivacao
    }

    dados_pre = preprocessar_dados(dados)
    dados_feat = transformar_features(dados_pre)
    modelo = carregar_modelo()
    resultado = fazer_inferencia(modelo, dados_feat)

    st.success(f"✅ Compatibilidade com a vaga: {resultado}")