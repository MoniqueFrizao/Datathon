
import pandas as pd
import os
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Caminho para os arquivos
data_path = r'C:\Users\monique_sandoval\decision-mvp\data'

# Funções de carregamento e normalização
def preprocess_vagas(file_path):
    raw = pd.read_json(file_path)
    vagas = []
    for col in raw.columns:
        vaga = {
            'vaga_id': col,
            **raw[col]['informacoes_basicas'],
            **raw[col]['perfil_vaga'],
            **raw[col]['beneficios']
        }
        vagas.append(vaga)
    return pd.DataFrame(vagas)

def preprocess_prospects(file_path):
    raw = pd.read_json(file_path)
    prospects = []
    for col in raw.columns:
        vaga_id = col
        prospect_list = raw[col]['prospects']
        for prospect in prospect_list:
            prospect['vaga_id'] = vaga_id
            prospects.append(prospect)
    return pd.DataFrame(prospects)

def preprocess_applicants(file_path):
    with open(file_path, encoding='utf-8') as f:
        raw_json = json.load(f)
    applicants = []
    for key, value in raw_json.items():
        applicant = {'applicant_id': key}
        applicant.update(value)
        applicants.append(applicant)
    return pd.DataFrame(applicants)

# Função de limpeza de texto
def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()



# Carregamento dos dados
df_vagas = preprocess_vagas(os.path.join(data_path, 'vagas.json'))
df_prospects = preprocess_prospects(os.path.join(data_path, 'prospects.json'))
df_applicants = preprocess_applicants(os.path.join(data_path, 'applicants.json'))

# Unir dados
df = df_prospects.merge(df_applicants, left_on='codigo', right_on='applicant_id', how='left')
df = df.merge(df_vagas, on='vaga_id', how='left')

# Criar colunas de texto
df['texto_vaga'] = df['principais_atividades'].fillna('') + ' ' + df['objetivo_vaga'].fillna('')
df['texto_cv'] = df['cv_pt'].fillna('').apply(clean_text)

# Vetorização TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['texto_vaga'].fillna('') + ' ' + df['titulo_vaga'].fillna(''))
tfidf_cv = vectorizer.transform(df['texto_cv'].fillna(''))

# Similaridade técnica
df['match_tecnico'] = [cosine_similarity(tfidf_cv[i], tfidf_matrix[i])[0][0] for i in range(tfidf_cv.shape[0])]

# Fit cultural
df['fit_cultural'] = df.apply(lambda row: 1 if isinstance(row.get('cidade'), str) and isinstance(row.get('infos_basicas'), dict) and row['cidade'].lower() in str(row['infos_basicas']).lower() else 0, axis=1)

# Engajamento
def score_engajamento(situacao):
    if not isinstance(situacao, str):
        return 0
    situacao = situacao.lower()
    if 'contratado' in situacao:
        return 1.0
    elif 'encaminhado' in situacao:
        return 0.7
    elif 'reprovado' in situacao:
        return 0.3
    else:
        return 0.5

df['score_engajamento'] = df['situacao_candidado'].apply(score_engajamento)

# Score final
df['score_final'] = 0.5 * df['match_tecnico'] + 0.3 * df['fit_cultural'] + 0.2 * df['score_engajamento']

# Resultado final
df_resultado = df[['nome_x', 'codigo', 'vaga_id', 'titulo_vaga', 'match_tecnico', 'fit_cultural', 'score_engajamento', 'score_final']]


def transformar_features(codigo_candidato):
    df = pd.read_csv(os.path.join('data', 'df_final.csv'))
    linha = df[df['codigo'] == codigo_candidato]
    if linha.empty:
        return None
    return linha[['match_tecnico', 'fit_cultural', 'score_engajamento']].values[0]

# Salvar resultado
output_path = os.path.join(data_path, 'df_final.csv')
df_resultado.to_csv(output_path, index=False)
print(f"\nArquivo df_final.csv salvo em: {output_path}")
