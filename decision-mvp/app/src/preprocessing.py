
import pandas as pd
import os
import json

# Caminho para os arquivos
data_path = r'C:\Users\monique_sandoval\decision-mvp\data'

def preprocess_vagas(file_path):
    raw = pd.read_json(file_path)
    vagas = []
    for col in raw.columns:
        vaga = {
            'id': col,
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
        applicant = {'id': key}
        applicant.update(value)
        applicants.append(applicant)
    return pd.DataFrame(applicants)

# Caminhos dos arquivos
vagas_file = os.path.join(data_path, 'vagas.json')
prospects_file = os.path.join(data_path, 'prospects.json')
applicants_file = os.path.join(data_path, 'applicants.json')

# Executa o pré-processamento
df_vagas = preprocess_vagas(vagas_file)
df_prospects = preprocess_prospects(prospects_file)
df_applicants = preprocess_applicants(applicants_file)

# Exibe amostras
print("\nVagas:")
print(df_vagas.head())

print("\nProspects:")
print(df_prospects.head())

print("\nApplicants:")
print(df_applicants.head())

print("\nColunas disponíveis em df_vagas:")
print(df_vagas.columns.tolist())
