

import joblib
import os

def carregar_modelo(caminho=None):
    if caminho is None:
        caminho = os.path.join(os.path.dirname(__file__), '..', 'models', 'modelo.pkl')
        caminho = os.path.abspath(caminho)
    return joblib.load(caminho)

def fazer_inferencia(modelo, dados):
    return modelo.predict([dados])[0]

