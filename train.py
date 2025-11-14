import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Charger les données
df = pd.read_csv("ecg_data.csv")

# Séparer features et labels
X = df.iloc[:, :-1].values.astype(np.float32)
y = df.iloc[:, -1].astype(int).values

# Normalisation
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Définition du modèle
model = Sequential([
    Dense(64, activation='sigmoid', input_shape=(X_train.shape[1],)),
    Dense(32, activation='sigmoid'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Entraînement
history = model.fit(X_train, y_train, epochs=10, batch_size=4, validation_split=0.1)

# Évaluation
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.4f}, Loss: {loss:.4f}")

# Sauvegarde du modèle et scaler
model_json = model.to_json()
model_weights = model.get_weights()

save_obj = {
    "model_json": model_json,
    "model_weights": model_weights,
    "scaler": scaler
}
with open("ecg_model.pkl", "wb") as f:
    pickle.dump(save_obj, f)

print("✅ Modèle sauvegardé sous ecg_model.pkl")
