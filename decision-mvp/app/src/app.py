


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


import streamlit as st
from model_utils import carregar_modelo, fazer_inferencia
from feature_engineering import transformar_features


# Configuração da página
st.set_page_config(page_title="Decision Match IA", layout="centered")

# Título e descrição
st.title("🔍 IA para Recrutamento - Decision")
st.markdown("### 🧠 Avaliação Inteligente de Candidatos")
st.markdown("Preencha os dados abaixo para calcular o **score de compatibilidade** com a vaga.")

# Layout em colunas
col1, col2 = st.columns(2)

with col1:
    nome = st.text_input("👤 Nome do candidato")
    idade = st.number_input("🎂 Idade", min_value=18, max_value=99)
    experiencia = st.selectbox("📊 Experiência (anos)", [0, 1, 2, 3, 5, 10])

with col2:
    linguagem = st.multiselect("💻 Linguagens que domina", ["Python", "Java", "C#", "JavaScript", "SQL"])
    motivacao = st.slider("🔥 Motivação para a vaga", 0, 10)

# Avaliação
if st.button("🚀 Avaliar Compatibilidade"):
    dados = {
        "nome": nome,
        "idade": idade,
        "experiencia": experiencia,
        "linguagem": linguagem,
        "motivacao": motivacao
    }

    # Aqui você pode adaptar para usar o código do candidato, se necessário
    codigo_candidato = st.text_input("🔢 Código do candidato (para buscar no df_final.csv)")
    if codigo_candidato:
        features = transformar_features(int(codigo_candidato))
        if features is None:
            st.error("❌ Candidato não encontrado no banco de dados.")
        else:
            modelo = carregar_modelo()
            resultado = fazer_inferencia(modelo, features)

            if resultado >= 0.8:
                st.success(f"✅ Excelente compatibilidade: {resultado:.2f}")
            elif resultado >= 0.5:
                st.warning(f"⚠️ Compatibilidade moderada: {resultado:.2f}")
            else:
                st.error(f"❌ Baixa compatibilidade: {resultado:.2f}")

# Rodapé
st.markdown("---")
st.markdown("📁 Projeto desenvolvido para o **Datathon Decision** | 💡 Por *Monique Frizao Guardia Sandoval*")
