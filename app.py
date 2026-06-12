import streamlit as st
import pandas as pd
import json
from groq import Groq

# Titre
st.title("🍷 SipMyGrape : Ton Sommelier IA")

# Charger les données
@st.cache_data
def load_data():
    with open('dataset.json', 'r', encoding='utf-8') as f:
        return json.load(f)

data = load_data()
df = pd.DataFrame(data)

# Zone de chat
user_input = st.text_input("Posez votre question sur les vins :")

if user_input:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    contexte = json.dumps(data, ensure_ascii=False)
    contexte_court = contexte[:15000] # Augmenté légèrement pour tes futurs 200 posts
    
    with st.spinner('Le sommelier analyse...'):
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "Tu es un sommelier expert. Ta réponse doit être scindée en deux parties strictes.\n\n"
                        "### 🍷 Dans ma cave\n"
                        "Propose 2 vins issus UNIQUEMENT de mes données qui correspondent à la question.\n"
                        "Format : '- [Nom] (Domaine) : [1 phrase max sur le profil]'\n\n"
                        "### 🌟 Suggestion Sommelier\n"
                        "Propose 1 vin extérieur à mes données, obligatoirement différent de ceux cités au-dessus.\n"
                        "Format : '- [Nom du vin] : [Pourquoi ce choix complète mes goûts + prix]'\n\n"
                        "RÈGLE D'OR : Ne jamais citer un vin dans la section 'Suggestion Sommelier' s'il est déjà dans 'Dans ma cave'.\n"
                        "Données de la cave : " + contexte_court
                    )
                },
                {"role": "user", "content": user_input}
            ]
        )
    
    st.markdown("---")
    st.write(completion.choices[0].message.content)
