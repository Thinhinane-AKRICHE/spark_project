import streamlit as st
import tempfile
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from predict import predire_rf, predire_xgb

# CONFIGURATION DE LA PAGE
st.set_page_config(
    page_title="Classification audio - Spark",
    page_icon="🔊",
    layout="wide"
)

# DONNÉES SUR LES 10 CLASSES DU DATASET UrbanSound8K
# Chemin vers un exemple audio pour chaque classe (relatif au projet)

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "Dataset_Sorted_by_class")

CLASSES_INFO = {
    "air_conditioner": {
        "emoji": "❄️",
        "description": "Son continu et régulier produit par un climatiseur.",
        "exemple": "13230-0-0-1.wav",
    },
    "car_horn": {
        "emoji": "📯",
        "description": "Klaxon de voiture, son court et intense.",
        "exemple": "100648-1-0-0.wav",
    },
    "children_playing": {
        "emoji": "🧒",
        "description": "Bruits de jeux d'enfants, sons variés et dynamiques.",
        "exemple": "178520-2-0-11.wav",
    },
    "dog_bark": {
        "emoji": "🐕",
        "description": "Aboiements de chien, sons impulsifs répétés.",
        "exemple": "109711-3-2-4.wav",
    },
    "drilling": {
        "emoji": "🔩",
        "description": "Perceuse ou foreuse, son mécanique à haute fréquence.",
        "exemple": "137815-4-0-10.wav",
    },
    "engine_idling": {
        "emoji": "🚗",
        "description": "Moteur tournant au ralenti, son grave et régulier.",
        "exemple": "144068-5-0-10.wav",
    },
    "gun_shot": {
        "emoji": "💥",
        "description": "Coup de feu, son très court et impulsif.",
        "exemple": "161195-6-0-0.wav",
    },
    "jackhammer": {
        "emoji": "🏗️",
        "description": "Marteau-piqueur, son répétitif et percussif.",
        "exemple": "14772-7-0-0.wav",
    },
    "siren": {
        "emoji": "🚨",
        "description": "Sirène de véhicule d'urgence, son oscillant.",
        "exemple": "157867-8-0-24.wav",
    },
    "street_music": {
        "emoji": "🎵",
        "description": "Musique de rue, sons mélodiques et rythmiques.",
        "exemple": "155242-9-0-35.wav",
    },
}

# INITIALISATION DE L'HISTORIQUE DE SESSION
# st.session_state persiste les données entre les reruns Streamlit
# On initialise la liste des prédictions si elle n'existe pas encore

if "historique" not in st.session_state:
    st.session_state.historique = []

# NAVIGATION PAR ONGLETS
# st.tabs crée plusieurs onglets dans la même page
onglet_prediction, onglet_historique, onglet_classes = st.tabs([
    "🎙️ Prédiction",
    "📋 Historique",
    "📖 Les classes"
])


# FONCTIONS UTILITAIRES

def couleur_confiance(score):
    """
    Retourne une couleur selon le niveau de confiance du modèle :
    - vert   si confiance >= 70%
    - orange si confiance entre 40% et 70%
    - rouge  si confiance < 40%
    """
    if score >= 0.70:
        return "green"
    elif score >= 0.40:
        return "orange"
    else:
        return "red"

def afficher_waveform_et_spectrogramme(file_path):
    """
    Charge le fichier audio avec librosa et affiche :
    1. La forme d'onde (amplitude en fonction du temps)
    2. Le spectrogramme mel (représentation temps/fréquence en dB)
       C'est cette représentation qui est indirectement utilisée par le modèle
       via les features mel_spec_mean/std.
    """
    y, sr = librosa.load(file_path, sr=None)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3))

    # --- Waveform ---
    axes[0].set_title("Forme d'onde")
    librosa.display.waveshow(y, sr=sr, ax=axes[0], color="#2196F3")
    axes[0].set_xlabel("Temps (s)")
    axes[0].set_ylabel("Amplitude")

    # --- Spectrogramme mel ---
    axes[1].set_title("Spectrogramme Mel")
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=axes[1])
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def afficher_barplot(probas, classe, titre=""):
    """
    Affiche un graphique en barres horizontales des probabilités par classe.
    La barre de la classe prédite est colorée en vert, les autres en bleu.
    """
    classes = list(probas.keys())
    valeurs = list(probas.values())
    couleurs = ["#4CAF50" if c == classe else "#2196F3" for c in classes]

    fig, ax = plt.subplots(figsize=(6, 4))
    if titre:
        ax.set_title(titre, fontsize=11)
    ax.barh(classes, valeurs, color=couleurs)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probabilité")
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def afficher_top3(probas):
    """
    Affiche les 3 classes les plus probables avec leur score,
    triées par probabilité décroissante.
    """
    top3 = sorted(probas.items(), key=lambda x: x[1], reverse=True)[:3]
    st.markdown("**Top 3 classes :**")
    for i, (cls, prob) in enumerate(top3):
        medaille = ["🥇", "🥈", "🥉"][i]
        st.write(f"{medaille} **{cls}** — {prob*100:.1f}%")


# ONGLET 1 : PRÉDICTION
with onglet_prediction:
    st.title("Classification de sons urbains")
    st.write("Upload un fichier audio pour prédire sa classe avec le modèle de ton choix.")

    # --- Sélection du modèle ---
    # st.selectbox crée un menu déroulant
    modele = st.selectbox(
        "Choisir le modèle de prédiction",
        ["Random Forest (Spark MLlib)", "XGBoost"]
    )

    # Checkbox pour activer la comparaison des deux modèles en parallèle
    comparer = st.checkbox("Comparer les deux modèles côte à côte")

    file = st.file_uploader("Choisir un fichier audio", type=["wav", "mp3", "ogg"])

    if file is not None:
        st.audio(file, format=file.type)

        # Sauvegarde temporaire du fichier uploadé sur le disque
        # (nécessaire car librosa et PySpark travaillent avec des chemins fichiers)
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        # --- Visualisation audio ---
        st.subheader("Visualisation du signal")
        afficher_waveform_et_spectrogramme(tmp_path)

        # ---------------------------------------------------------------
        # MODE COMPARAISON : on lance les deux modèles et on les affiche
        # côte à côte avec st.columns
        # ---------------------------------------------------------------
        if comparer:
            st.subheader("Comparaison des deux modèles")
            with st.spinner("Analyse avec les deux modèles..."):
                col_rf, col_xgb = st.columns(2)

                with col_rf:
                    st.markdown("### Random Forest")
                    classe_rf, probas_rf = predire_rf(tmp_path)
                    confiance_rf = probas_rf[classe_rf]
                    couleur_rf = couleur_confiance(confiance_rf)
                    st.markdown(
                        f"**Classe :** {classe_rf}  \n"
                        f"**Confiance :** :{couleur_rf}[{confiance_rf*100:.1f}%]"
                    )
                    afficher_top3(probas_rf)
                    afficher_barplot(probas_rf, classe_rf)

                with col_xgb:
                    st.markdown("### XGBoost")
                    classe_xgb, probas_xgb = predire_xgb(tmp_path)
                    confiance_xgb = probas_xgb[classe_xgb]
                    couleur_xgb = couleur_confiance(confiance_xgb)
                    st.markdown(
                        f"**Classe :** {classe_xgb}  \n"
                        f"**Confiance :** :{couleur_xgb}[{confiance_xgb*100:.1f}%]"
                    )
                    afficher_top3(probas_xgb)
                    afficher_barplot(probas_xgb, classe_xgb)

            # Ajout dans l'historique des deux prédictions
            st.session_state.historique.append({
                "Fichier": file.name,
                "Modèle": "Random Forest",
                "Classe prédite": classe_rf,
                "Confiance": f"{probas_rf[classe_rf]*100:.1f}%",
            })
            st.session_state.historique.append({
                "Fichier": file.name,
                "Modèle": "XGBoost",
                "Classe prédite": classe_xgb,
                "Confiance": f"{probas_xgb[classe_xgb]*100:.1f}%",
            })

        # MODE SIMPLE : on lance uniquement le modèle sélectionné
        else:
            st.subheader("Résultat de la prédiction")
            with st.spinner("Analyse en cours..."):
                if modele == "Random Forest (Spark MLlib)":
                    classe, probas = predire_rf(tmp_path)
                else:
                    classe, probas = predire_xgb(tmp_path)

            confiance = probas[classe]
            couleur = couleur_confiance(confiance)

            # Affichage de la classe prédite et du score de confiance coloré
            st.markdown(
                f"### Classe prédite : **{classe}**  \n"
                f"Confiance : :{couleur}[{confiance*100:.1f}%]"
            )

            # Top 3 + barplot côte à côte
            col_top3, col_graph = st.columns([1, 2])
            with col_top3:
                afficher_top3(probas)
            with col_graph:
                afficher_barplot(probas, classe)

            # Ajout de la prédiction dans l'historique de session
            st.session_state.historique.append({
                "Fichier": file.name,
                "Modèle": modele,
                "Classe prédite": classe,
                "Confiance": f"{confiance*100:.1f}%",
            })

        # Suppression du fichier temporaire après traitement
        os.remove(tmp_path)


# ONGLET 2 : HISTORIQUE DES PRÉDICTIONS
# Affiche toutes les prédictions faites pendant la session en cours
with onglet_historique:
    st.title("Historique des prédictions")

    if not st.session_state.historique:
        st.info("Aucune prédiction effectuée pour l'instant.")
    else:
        # st.dataframe affiche un tableau interactif
        st.dataframe(
            st.session_state.historique,
            use_container_width=True
        )
        # Bouton pour réinitialiser l'historique
        if st.button("Effacer l'historique"):
            st.session_state.historique = []
            st.rerun()


# ONGLET 3 : PRÉSENTATION DES 10 CLASSES
# Pour chaque classe, on affiche sa description et un exemple audio jouable
with onglet_classes:
    st.title("Les 10 classes du dataset UrbanSound8K")
    st.write("Exemples audio de chaque classe utilisée pour entraîner les modèles.")

    # Affichage en grille : 2 colonnes par ligne
    classes_liste = list(CLASSES_INFO.items())
    for i in range(0, len(classes_liste), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j >= len(classes_liste):
                break
            nom, info = classes_liste[i + j]
            exemple_path = os.path.join(DATASET_DIR, nom, info["exemple"])
            with col:
                st.markdown(f"#### {info['emoji']} {nom.replace('_', ' ').title()}")
                st.write(info["description"])
                # Lecture de l'exemple audio s'il existe sur le disque
                if os.path.exists(exemple_path):
                    with open(exemple_path, "rb") as f:
                        st.audio(f.read(), format="audio/wav")
                else:
                    st.caption("Exemple audio non disponible.")