import streamlit as st
import tempfile
import os
import matplotlib.pyplot as plt
from predict import predire

st.set_page_config(
    page_title="Classification audio - spark", 
    page_icon="🔊",
    layout="centered")

st.title("Classification de sons urbains")
st.write("Upload un fichier audio et le modèle Random Forest Spark prédit sa classe ")

file = st.file_uploader("Choisir un fichier audio", type=["wav", "mp3", "ogg"])

if file is not None:
    st.audio(file, format=file.type)
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
        tmp_file.write(file.read())
        tmp_file_path = tmp_file.name
    with st.spinner("Analyse en cours..."):  
        classe, probas = predire(tmp_file_path)

    os.remove(tmp_file_path)
    st.subheader(f"Prrobabilité de classe : {classe}")  
    classes = list(probas.keys())
    valeurs = list(probas.values())

    fig, ax = plt.subplots(figsize=(8, 4))
    couleurs = ["#2196F3" if c != classe else "#4CAF50" for c in classes]
    ax.barh(classes, valeurs, color=couleurs)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probabilité")
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)