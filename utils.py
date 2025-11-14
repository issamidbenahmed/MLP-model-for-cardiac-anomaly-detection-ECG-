import pickle
from tensorflow.keras.models import model_from_json

def load_model_from_pkl(path="ecg_model.pkl"):
    # Charger le fichier pickle
    with open(path, "rb") as f:
        data = pickle.load(f)

    # Reconstruire le modèle à partir du JSON
    model = model_from_json(data["model_json"])
    model.set_weights(data["model_weights"])
    scaler = data["scaler"]

    # Compiler le modèle (obligatoire pour faire des prédictions)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model, scaler
