import os
import re
import json
import pickle
import requests
import numpy as np
import streamlit as st

# ── Configuration page ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Détecteur de fausses offres d'emploi",
    page_icon="🔍",
    layout="wide",
)

# ── Chargement des modèles (mis en cache) ────────────────────────────────────
@st.cache_resource(show_spinner="Chargement du modèle BiLSTM…")
def load_lstm():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences  # noqa

    model = tf.keras.models.load_model("lstm_model.keras")
    with open("keras_tokenizer.pkl", "rb") as f:
        tok = pickle.load(f)
    return model, tok


@st.cache_resource(show_spinner="Chargement de BART-large-mnli…")
def load_hf():
    from transformers import pipeline as hf_pipeline

    return hf_pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1,
        framework="pt",
    )


LABELS_HF = ["legitimate job posting", "fake fraudulent job posting"]
MAX_LEN = 300


# ── Fonctions de prédiction ───────────────────────────────────────────────────
def preprocess(title, company, description, requirements, benefits):
    text = " ".join([title, company, description, requirements, benefits])
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_classic(text):
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    model, tok = load_lstm()
    seq = pad_sequences(tok.texts_to_sequences([text]), maxlen=MAX_LEN, truncating="post")
    prob = float(model.predict(seq, verbose=0)[0][0])
    return {
        "label": "🚨 Fausse" if prob > 0.5 else "✅ Réelle",
        "is_fake": prob > 0.5,
        "prob": prob,
        "confidence": prob if prob > 0.5 else 1 - prob,
    }


def predict_opensource(text):
    clf = load_hf()
    result = clf(text[:512], candidate_labels=LABELS_HF)
    is_fake = result["labels"][0] == LABELS_HF[1]
    scores = dict(zip(result["labels"], result["scores"]))
    return {
        "label": "🚨 Fausse" if is_fake else "✅ Réelle",
        "is_fake": is_fake,
        "confidence": result["scores"][0],
        "score_fake": scores[LABELS_HF[1]],
        "score_real": scores[LABELS_HF[0]],
    }


def predict_api(text, api_key):
    prompt = (
        "Analyze this job posting and classify it as FAKE or REAL.\n\n"
        f'Job Posting:\n"""\n{text[:800]}\n"""\n\n'
        "Red flags for FAKE: vague description, unrealistic salary, requests personal info, "
        "poor grammar, no company details.\n\n"
        'Respond ONLY with valid JSON: {"prediction": "FAKE" or "REAL", '
        '"confidence": 0.0-1.0, "reason": "one sentence"}'
    )
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "openai/gpt-oss-120b:free",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert HR fraud analyst detecting fake job postings.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 150,
                "temperature": 0.1,
            },
            timeout=30,
        )
        content = resp.json()["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            parsed = json.loads(m.group())
            pred = parsed.get("prediction", "REAL").upper()
            return {
                "label": "🚨 Fausse" if pred == "FAKE" else "✅ Réelle",
                "is_fake": pred == "FAKE",
                "confidence": float(parsed.get("confidence", 0.5)),
                "reason": parsed.get("reason", ""),
            }
    except Exception as e:
        return {"label": "❌ Erreur", "is_fake": None, "confidence": 0.0, "reason": str(e)}
    return {"label": "❌ Erreur", "is_fake": None, "confidence": 0.0, "reason": "Réponse non parseable"}


# ── Interface ─────────────────────────────────────────────────────────────────
st.title("🔍 Détecteur de Fausses Offres d'Emploi")
st.caption("Backend Intelligent — Comparaison de 3 approches IA")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input(
        "Clé OpenRouter (API IA)",
        value=os.getenv("OPENROUTER_API_KEY", ""),
        type="password",
        help="Obtenez une clé gratuite sur openrouter.ai",
    )
    st.divider()
    st.markdown("**Approches utilisées :**")
    st.markdown("1. 🧠 **BiLSTM** — entraîné from scratch")
    st.markdown("2. 🤗 **BART-large-mnli** — zero-shot HuggingFace")
    st.markdown("3. 🌐 **Mistral-7B** — via OpenRouter API")
    st.divider()
    st.markdown("**Exemples rapides :**")
    col_ex1, col_ex2 = st.columns(2)
    load_fake = col_ex1.button("Charger fausse", use_container_width=True)
    load_real = col_ex2.button("Charger réelle", use_container_width=True)

# Exemples pré-remplis
EXAMPLE_FAKE = {
    "title": "Travail à domicile — Gagnez 5000€/semaine sans expérience !",
    "company": "",
    "description": (
        "Opportunité incroyable ! Travaillez depuis chez vous, aucune expérience requise. "
        "Revenus garantis de 5000€/semaine. Envoyez vos coordonnées bancaires et une avance "
        "de 50€ pour démarrer immédiatement. Places limitées !"
    ),
    "requirements": "Aucune qualification nécessaire. Avoir un compte bancaire.",
    "benefits": "Revenus illimités, liberté totale, travail depuis partout dans le monde.",
}

EXAMPLE_REAL = {
    "title": "Ingénieur Backend Python — CDI",
    "company": (
        "Acme Corp est une société fintech basée à Paris, 250 employés, fondée en 2015. "
        "Nous développons des solutions de paiement en ligne pour les PME européennes."
    ),
    "description": (
        "Nous recherchons un ingénieur backend expérimenté pour rejoindre notre équipe technique. "
        "Vous participerez au développement de notre API REST, à l'optimisation des performances "
        "et à la mise en place de bonnes pratiques CI/CD."
    ),
    "requirements": (
        "3+ ans d'expérience Python. Maîtrise de FastAPI ou Django. "
        "Connaissance de PostgreSQL et Redis. Bac+5 en informatique ou équivalent."
    ),
    "benefits": "Mutuelle Alan, 2 jours de télétravail/semaine, stock-options, RTT.",
}

# Gestion des exemples
defaults = {"title": "", "company": "", "description": "", "requirements": "", "benefits": ""}
if load_fake:
    defaults = EXAMPLE_FAKE
    st.session_state.update(defaults)
if load_real:
    defaults = EXAMPLE_REAL
    st.session_state.update(defaults)

# Formulaire de saisie
st.subheader("📋 Saisir l'offre d'emploi")
@st.cache_data(show_spinner=False)
def get_job_titles():
    import pandas as pd
    df = pd.read_csv("fake_job_postings.csv")
    titles = sorted(df["title"].dropna().unique().tolist())
    return ["— Saisie libre —"] + titles


with st.form("job_form"):
    all_titles = get_job_titles()
    selected = st.selectbox(
        "Titre du poste *",
        options=all_titles,
        index=all_titles.index(st.session_state.get("title", "— Saisie libre —"))
        if st.session_state.get("title", "— Saisie libre —") in all_titles
        else 0,
    )
    if selected == "— Saisie libre —":
        title = st.text_input(
            "Saisir un titre personnalisé",
            value=st.session_state.get("title", defaults["title"])
            if st.session_state.get("title", "") not in all_titles
            else "",
            placeholder="ex : Développeur Full Stack — CDI Paris",
        )
    else:
        title = selected
    col1, col2 = st.columns(2)
    with col1:
        company = st.text_area(
            "Profil de l'entreprise",
            value=st.session_state.get("company", defaults["company"]),
            height=100,
            placeholder="Présentation de la société…",
        )
        requirements = st.text_area(
            "Compétences requises",
            value=st.session_state.get("requirements", defaults["requirements"]),
            height=100,
            placeholder="Expériences, diplômes, compétences techniques…",
        )
    with col2:
        description = st.text_area(
            "Description du poste *",
            value=st.session_state.get("description", defaults["description"]),
            height=100,
            placeholder="Missions, responsabilités…",
        )
        benefits = st.text_area(
            "Avantages",
            value=st.session_state.get("benefits", defaults["benefits"]),
            height=100,
            placeholder="Salaire, avantages en nature…",
        )

    submitted = st.form_submit_button("🔍 Analyser l'offre", use_container_width=True, type="primary")

# ── Résultats ─────────────────────────────────────────────────────────────────
if submitted:
    if not title and not description:
        st.warning("Veuillez au minimum renseigner le titre et la description.")
    else:
        text = preprocess(title, company, description, requirements, benefits)

        st.divider()
        st.subheader("📊 Résultats de l'analyse")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["🧠 IA Classique (BiLSTM)", "🤗 Open Source (HuggingFace)", "🌐 API (OpenRouter)", "⚖️ Comparaison"]
        )

        # ── Tab 1 : BiLSTM ────────────────────────────────────────────────────
        with tab1:
            with st.spinner("Analyse BiLSTM en cours…"):
                r1 = predict_classic(text)

            verdict_color = "red" if r1["is_fake"] else "green"
            st.markdown(
                f"<h2 style='text-align:center; color:{verdict_color}'>{r1['label']}</h2>",
                unsafe_allow_html=True,
            )
            col_a, col_b = st.columns(2)
            col_a.metric("Probabilité de fraude", f"{r1['prob']:.1%}")
            col_b.metric("Confiance", f"{r1['confidence']:.1%}")
            st.progress(r1["prob"], text=f"Score de fraude : {r1['prob']:.1%}")
            st.info(
                "**Modèle** : Bidirectional LSTM entraîné from scratch sur 17 880 offres réelles.\n\n"
                "**Avantages** : Meilleure accuracy (94.9%), AUC-ROC 0.96, rapide en inférence.\n\n"
                "**Limites** : Boîte noire, nécessite des données labellisées."
            )

        # ── Tab 2 : HuggingFace ───────────────────────────────────────────────
        with tab2:
            with st.spinner("Analyse BART zero-shot en cours (peut prendre ~10s)…"):
                r2 = predict_opensource(text)

            verdict_color = "red" if r2["is_fake"] else "green"
            st.markdown(
                f"<h2 style='text-align:center; color:{verdict_color}'>{r2['label']}</h2>",
                unsafe_allow_html=True,
            )
            col_a, col_b = st.columns(2)
            col_a.metric("Score — Fausse", f"{r2['score_fake']:.1%}")
            col_b.metric("Score — Réelle", f"{r2['score_real']:.1%}")
            st.progress(r2["score_fake"], text=f"Probabilité fausse : {r2['score_fake']:.1%}")
            st.info(
                "**Modèle** : facebook/bart-large-mnli — classification zero-shot via NLI.\n\n"
                "**Avantages** : Aucune donnée d'entraînement requise, facilement adaptable.\n\n"
                "**Limites** : Lent sur CPU, sensible au déséquilibre des classes sans fine-tuning."
            )

        # ── Tab 3 : OpenRouter ────────────────────────────────────────────────
        with tab3:
            if not api_key:
                st.warning("Clé OpenRouter non renseignée. Ajoutez-la dans la barre latérale.")
            else:
                with st.spinner("Appel API Mistral-7B en cours…"):
                    r3 = predict_api(text, api_key)

                verdict_color = "red" if r3["is_fake"] else ("green" if r3["is_fake"] is False else "gray")
                st.markdown(
                    f"<h2 style='text-align:center; color:{verdict_color}'>{r3['label']}</h2>",
                    unsafe_allow_html=True,
                )
                if r3["is_fake"] is not None:
                    st.metric("Confiance", f"{r3['confidence']:.1%}")
                    st.progress(r3["confidence"])
                if r3.get("reason"):
                    st.markdown(f"> 💬 **Explication** : {r3['reason']}")
                st.info(
                    "**Modèle** : openai/gpt-oss-120b:free via OpenRouter API.\n\n"
                    "**Avantages** : Explication en langage naturel, aucune ressource locale.\n\n"
                    "**Limites** : Latence réseau, limites de requêtes sur le tier gratuit."
                )

        # ── Tab 4 : Comparaison ───────────────────────────────────────────────
        with tab4:
            st.markdown("### Synthèse des prédictions")

            results = {"BiLSTM": r1}
            if "r2" in dir():
                results["BART ZS"] = r2
            if api_key and "r3" in dir() and r3["is_fake"] is not None:
                results["Mistral-7B"] = r3

            rows = []
            votes_fake = 0
            votes_real = 0
            for nom, res in results.items():
                verdict = res["label"]
                conf = res.get("confidence", res.get("prob", 0.5))
                rows.append({"Modèle": nom, "Verdict": verdict, "Confiance": f"{conf:.1%}"})
                if res["is_fake"]:
                    votes_fake += 1
                else:
                    votes_real += 1

            import pandas as pd
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.divider()
            total = votes_fake + votes_real
            ensemble = "🚨 FAUSSE" if votes_fake > votes_real else "✅ RÉELLE"
            color = "red" if votes_fake > votes_real else "green"
            st.markdown(
                f"<h3 style='text-align:center'>Vote majoritaire : "
                f"<span style='color:{color}'>{ensemble}</span> "
                f"({max(votes_fake, votes_real)}/{total} modèles)</h3>",
                unsafe_allow_html=True,
            )

            with st.expander("📈 Performances générales des modèles (sur jeu de test)"):
                perf = pd.DataFrame([
                    {"Modèle": "BiLSTM (IA Classique)", "Accuracy": "94.91%", "AUC-ROC": "0.9617",
                     "F1 Fausse": "0.606", "Jeu d'éval": "3 576 offres"},
                    {"Modèle": "BART Zero-Shot (HuggingFace)", "Accuracy": "94.67%", "AUC-ROC": "0.552",
                     "F1 Fausse": "0.000", "Jeu d'éval": "150 échantillons"},
                    {"Modèle": "GPT-OSS-120B (OpenRouter)", "Accuracy": "62.5%", "AUC-ROC": "N/A",
                     "F1 Fausse": "0.400", "Jeu d'éval": "8 échantillons"},
                ])
                st.dataframe(perf, use_container_width=True, hide_index=True)
                st.caption(
                    "⚠️ L'accuracy élevée de BART (94.67%) est trompeuse : "
                    "le dataset est très déséquilibré (~95% d'offres réelles). "
                    "Le F1=0 montre qu'il ne détecte aucune fausse offre sans fine-tuning."
                )
