# Détecteur de Fausses Offres d'Emploi

Application web Streamlit qui détecte les offres d'emploi frauduleuses en combinant trois approches d'intelligence artificielle complémentaires.

## Démonstration

Renseignez une offre d'emploi (titre, profil de l'entreprise, description, compétences requises, avantages) et obtenez instantanément les prédictions des trois modèles ainsi qu'un verdict final par vote majoritaire.

## Approches comparées

| Modèle | Type | Accuracy | AUC-ROC |
|--------|------|----------|---------|
| **BiLSTM** (entraîné from scratch) | Deep learning classique | 94,91 % | 0,9617 |
| **BART-large-mnli** (HuggingFace) | Zero-shot NLI | 94,67 %* | 0,552 |
| **GPT-OSS-120B** (API OpenRouter) | LLM avec explication | 62,5 % | N/A |

> *La haute accuracy de BART est trompeuse — le dataset contient ~95 % d'offres réelles (déséquilibre de classes). Son F1=0,00 sur les fausses offres montre qu'il n'en détecte aucune sans fine-tuning.

## Jeu de données

- **Source** : [Employment Scam Aegean Dataset (EMSCAD)](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- **Taille** : 17 880 offres d'emploi (~4,8 % frauduleuses)
- **Fichier** : `fake_job_postings.csv`

## Structure du projet

```
├── app.py                  # Application Streamlit
├── main.py                 # Script d'entraînement du BiLSTM
├── fakejob.ipynb           # Analyse exploratoire et comparaison des modèles
├── lstm_model.keras        # Poids du modèle BiLSTM entraîné
├── keras_tokenizer.pkl     # Tokenizer Keras ajusté
├── fake_job_postings.csv   # Dataset brut
├── requirements.txt        # Dépendances Python
├── class_distribution.png  # Répartition des classes dans le dataset
├── lstm_training.png       # Courbes d'entraînement du BiLSTM
├── lstm_cm.png             # Matrice de confusion du BiLSTM
├── model_comparison.png    # Comparaison visuelle des modèles
└── model_comparison.csv    # Résultats numériques de la comparaison
```

## Installation

```bash
git clone https://github.com/Yacineismael/Fake-job-posting-.git
cd Fake-job-posting-
pip install -r requirements.txt
```

## Lancement

```bash
streamlit run app.py
```

Ouvrez ensuite [http://localhost:8501](http://localhost:8501) dans votre navigateur.

Pour utiliser l'onglet OpenRouter (GPT), ajoutez votre clé API gratuite depuis [openrouter.ai](https://openrouter.ai) dans la barre latérale.

## Détails des modèles

### BiLSTM (IA Classique)
- Architecture : Embedding → LSTM Bidirectionnel → Dense
- Entrée : concaténation de tous les champs textuels, mis en minuscule et nettoyés du HTML
- Longueur maximale de séquence : 300 tokens
- Jeu d'entraînement : 14 304 offres / Jeu de test : 3 576 offres

### BART Zero-Shot (HuggingFace)
- Modèle : `facebook/bart-large-mnli`
- Labels : `"legitimate job posting"` vs `"fake fraudulent job posting"`
- Aucun entraînement requis — inférence NLI pure

### GPT-OSS-120B (API OpenRouter)
- Retourne un verdict JSON structuré accompagné d'une explication en langage naturel
- Prompt conçu pour détecter les signaux d'alerte connus : description vague, salaire irréaliste, demande d'informations personnelles, fautes de grammaire, absence de détails sur l'entreprise

## Prérequis

- Python 3.9+
- TensorFlow 2.x
- PyTorch (pour l'inférence BART)
- Streamlit
- Transformers (HuggingFace)

Voir `requirements.txt` pour la liste complète.

## Auteur

**Yacine Ismail** — Mastère Data Intelligence Artificielle, NEXA
