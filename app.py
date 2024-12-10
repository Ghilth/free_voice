import streamlit as st
from bot_ask import ask  

# Configuration de la page
st.set_page_config(
    page_title="VoixLibre - Assistance",
    page_icon="💬",
    layout="centered"
)

# Appliquer un style personnalisé
st.markdown("""
    <style>
        body {
            background-color: #f2e8d3; /* Fond beige clair */
        }
        .stTextArea textarea {
            border: 2px solid #f2c277; /* Bordure assortie aux bulles */
            border-radius: 10px;
            font-size: 16px;
            padding: 10px;
        }
        .stButton button {
            background-color: #f2c277; /* Bouton assorti */
            color: white;
            border-radius: 5px;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
        }
        .stButton button:hover {
            background-color: #e5a960; /* Couleur plus foncée au survol */
        }
        .response-box {
            background-color: #ffffff; /* Couleur de la bulle Baké */
            padding: 15px;
            border-radius: 10px;
            color: black;
            font-size: 16px;
            margin: 10px 0;
        }
        .user-box {
            background-color: #f28d3; /* Couleur de la bulle utilisateur */
            padding: 15px;
            border-radius: 10px;
            color: black;
            font-size: 16px;
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

# Titre de l'application
st.title("VoixLibre 💬")

# Message d'accueil
st.subheader("👋 Bienvenue, je suis Baké !")
st.markdown("""
    Je suis là pour t'écouter, te guider, et te fournir des réponses claires à tes questions sur tes droits 
    et les options disponibles face aux violences basées sur le genre.  
    N’hésite pas à me poser tes questions en toute sérénité. Ensemble, nous trouverons des solutions. 🤝
""")

# Entrée utilisateur
user_input = st.text_area(
    "Posez votre question ici 👇",
    placeholder="Exemple : Quels sont mes droits en cas de harcèlement ?"
)

# Bouton pour envoyer la question
if st.button("Envoyer"):
    if user_input.strip():
        with st.spinner("Baké réfléchit à une réponse pour vous..."):
            # Appel de la fonction `ask` pour obtenir la réponse
            response = ask(user_input)
        # Affichage de la réponse
        st.markdown('<div class="user-box">' + user_input + '</div>', unsafe_allow_html=True)
        st.markdown('<div class="response-box">' + response + '</div>', unsafe_allow_html=True)
    else:
        st.warning("Veuillez entrer une question avant d'envoyer.")

# Ajout d'une note de confidentialité
st.markdown("""
    ---
    🔒 **Confidentialité garantie :** Toutes vos interactions avec moi, Baké, restent strictement confidentielles.
""")
