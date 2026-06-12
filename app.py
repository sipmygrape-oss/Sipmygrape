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
    
    # Conversion du JSON en texte propre
    contexte = json.dumps(data, ensure_ascii=False)
    contexte_court = contexte[:10000]
    
    with st.spinner('Le sommelier analyse tes goûts...'):
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "Tu es un sommelier expert. Tu as deux rôles : "
                        "1. Analyser les vins que l'utilisateur a bus (fournis dans les données ci-dessous). "
                        "2. Proposer des accords mets-vins et des conseils basés sur tes connaissances professionnelles, "
                        "en te basant PRIORITAIREMENT sur le style, le cépage et le profil des vins présents dans les données de l'utilisateur. "
                        "Si l'utilisateur demande un vin qui ne correspond pas du tout à ses habitudes, explique pourquoi. "
                        "Voici les vins bus par l'utilisateur : " + contexte_court
                    )
                },
                {"role": "user", "content": user_input}
            ]
        )
    
    st.write(completion.choices[0].message.content)
