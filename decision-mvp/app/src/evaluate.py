
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
import joblib
import os

# Caminhos
data_path = r'C:\Users\monique_sandoval\decision-mvp\data\df_final.csv'
modelo_path = r'C:\Users\monique_sandoval\decision-mvp\models\modelo.pkl'

# Carregar os dados
df = pd.read_csv(data_path)

# Features e target
features = ['match_tecnico', 'fit_cultural', 'score_engajamento']
target = 'score_final'

# Criar variável binária para classificação
df['aprovado'] = (df[target] >= 0.6).astype(int)

# Separar dados com estratificação
X = df[features]
y = df['aprovado']
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in sss.split(X, y):
    X_test = X.iloc[test_index]
    y_test = y.iloc[test_index]

# Carregar modelo treinado
model = joblib.load(modelo_path)

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliar desempenho
print("=== Avaliação do Modelo ===")
print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
