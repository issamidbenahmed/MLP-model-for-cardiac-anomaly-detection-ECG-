# 🫀 Prédiction d'Anomalies ECG

Application web de détection d'anomalies dans les signaux ECG (électrocardiogramme) utilisant un MLP (Multi-Layer Perceptron) et Flask.

## 📋 Description

Ce projet implémente un système de classification binaire pour détecter les anomalies dans les signaux ECG. Il comprend :
- Un modèle MLP (Multi-Layer Perceptron) entraîné avec TensorFlow/Keras
- Une API REST Flask pour les prédictions en temps réel
- Une interface web interactive pour tester le modèle
- Des Dockerfiles pour l'entraînement et le déploiement

## 🚀 Fonctionnalités

- **Entraînement du modèle** : Script Python pour entraîner un réseau de neurones sur des données ECG
- **API de prédiction** : Endpoint REST pour classifier les signaux ECG
- **Interface utilisateur** : Interface web simple et intuitive pour tester les prédictions
- **Containerisation** : Support Docker pour l'entraînement et le déploiement
- **Normalisation** : Prétraitement des données avec StandardScaler

## 🛠️ Technologies

- **Backend** : Flask, Python 3.10
- **Machine Learning** : TensorFlow 2.12, Keras
- **Prétraitement** : scikit-learn, pandas, numpy
- **Frontend** : HTML5, CSS3, JavaScript (Vanilla)
- **Containerisation** : Docker

## 📁 Structure du Projet

```
.
├── app.py                  # Application Flask (API + serveur web)
├── train.py                # Script d'entraînement du modèle
├── utils.py                # Fonctions utilitaires (chargement du modèle)
├── requirements.txt        # Dépendances Python
├── ecg_model.pkl          # Modèle entraîné (pickle)
├── ecg_data.csv           # Dataset ECG
├── Dockerfile.dep         # Dockerfile pour le déploiement
├── Dockerfile.train       # Dockerfile pour l'entraînement
└── templates/
    └── index.html         # Interface web
```

## 📊 Architecture du Modèle

Le modèle est un **MLP (Multi-Layer Perceptron)** avec l'architecture suivante :
- **Couche d'entrée** : 140 features (signaux ECG)
- **Couche cachée 1** : 64 neurones, activation sigmoid
- **Couche cachée 2** : 32 neurones, activation sigmoid
- **Couche de sortie** : 1 neurone, activation sigmoid (classification binaire)

**Type** : Fully Connected Neural Network (Dense layers)  
**Optimiseur** : Adam (learning rate = 0.001)  
**Fonction de perte** : Binary Crossentropy

## 🔧 Installation

### Prérequis

- Python 3.10+
- pip

### Installation locale

1. Cloner le repository
```bash
git clone https://github.com/issamidbenahmed/MLP-model-for-cardiac-anomaly-detection-ECG-.git

cd MLP-model-for-cardiac-anomaly-detection-ECG-
```

2. Installer les dépendances
```bash
pip install -r requirements.txt
```

3. Entraîner le modèle (optionnel)
```bash
python train.py
```

4. Lancer l'application
```bash
python app.py
```

L'application sera accessible sur `http://localhost:5000`

## 🐳 Utilisation avec Docker

### Entraîner le modèle avec Docker

```bash
docker build -f Dockerfile.train -t ecg-train .
docker run -v ${PWD}:/app ecg-train
```

### Déployer l'application avec Docker

```bash
docker build -f Dockerfile.dep -t ecg-app .
docker run -p 5000:5000 ecg-app
```

Accédez à l'application sur `http://localhost:5000`

## 📡 API

### Endpoint de prédiction

**POST** `/predict`

**Body (JSON)** :
```json
{
  "data": [0.1, -0.5, 1.2, ..., -0.2, 1.0]
}
```
*Note : Le tableau doit contenir exactement 140 valeurs numériques*

**Réponse** :
```json
{
  "probabilities": [0.8523],
  "predictions": [1]
}
```

- `probabilities` : Probabilité d'anomalie (0 à 1)
- `predictions` : Classe prédite (0 = normal, 1 = anomalie)

## 💻 Utilisation de l'Interface Web

1. Accédez à `http://localhost:5000`
2. Copiez une ligne de données ECG (140 valeurs séparées par des virgules)
3. Collez les données dans le champ de texte
4. Cliquez sur "Faire la prédiction"
5. Le résultat s'affiche avec :
   - ✅ Signal normal (classe 0)
   - ⚠️ Anomalie détectée (classe 1)
   - La probabilité associée

## 📈 Format des Données

Le dataset `ecg_data.csv` doit contenir :
- **140 colonnes** : Features extraites du signal ECG
- **1 colonne** : Label (0 = normal, 1 = anomalie)

Exemple de ligne :
```
0.1,-0.5,1.2,...,-0.2,1.0,1
```

## 🔍 Détails Techniques

### Prétraitement
- Normalisation avec `StandardScaler` (moyenne = 0, écart-type = 1)
- Split train/test : 80/20 avec stratification

### Entraînement
- Epochs : 10
- Batch size : 4
- Validation split : 10%

### Sauvegarde du Modèle
Le modèle est sauvegardé au format pickle avec :
- Architecture du modèle (JSON)
- Poids du modèle
- Scaler pour la normalisation

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou une pull request.


## 👤 Auteur

Aissam Id Ben Ahmed - [GitHub](https://github.com/issamidbenahmed)


