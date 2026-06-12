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
user_input = st.text_input("Pose ta question sur les vins :")

if user_input:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    
    # 1. On convertit tes données en texte propre (JSON formaté) au lieu d'un tableau
    contexte = json.dumps(data, ensure_ascii=False)
    
    # 2. On raccourcit le contexte pour éviter de dépasser la limite de l'IA
    # On prend les 10 000 premiers caractères pour être sûr que ça passe
    contexte_court = contexte[:10000]
    
    with st.spinner('Le sommelier réfléchit...'):
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Tu es un sommelier expert. Utilise ces données pour répondre : " + contexte_court},
                {"role": "user", "content": user_input}
            ]
        )
    
    st.write(completion.choices[0].message.content)
