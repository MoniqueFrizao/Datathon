
import pandas as pd
import os

# Caminho para os arquivos
data_path = r'C:\Users\monique_sandoval\decision-mvp\data'

# Arquivos JSON
files = {
    'vagas': 'vagas.json',
    'prospects': 'prospects.json',
    'applicants': 'applicants.json'
}

# Função para análise exploratória
def eda_json(file_path, name):
    print(f"\n{'='*40}\nAnálise do arquivo: {name}\n{'='*40}")
    try:
        df = pd.read_json(file_path)
        print(f"Dimensões: {df.shape}")
        print("\nPrimeiras linhas:")
        print(df.head())
        print("\nTipos de dados:")
        print(df.dtypes)
        print("\nValores nulos por coluna:")
        print(df.isnull().sum())
        print("\nEstatísticas básicas (numéricas):")
        print(df.describe(include='number'))
        print("\nEstatísticas básicas (categóricas):")
        print(df.describe(include='object'))
    except Exception as e:
        print(f"Erro ao processar {name}: {e}")

# Executa EDA para cada arquivo
for name, filename in files.items():
    file_path = os.path.join(data_path, filename)
    eda_json(file_path, name)
