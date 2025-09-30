import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier
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

# Separar dados
X = df[features]
y = df['aprovado']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar modelo
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Avaliação
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Salvar modelo
os.makedirs(os.path.dirname(modelo_path), exist_ok=True)
joblib.dump(model, modelo_path)
print(f"Modelo salvo em: {modelo_path}")