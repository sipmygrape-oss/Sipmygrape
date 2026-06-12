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
    contexte = df.to_string()
    
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": f"Voici mes posts Instagram : {contexte}. Réponds à : {user_input}"}]
    )
    
    st.write(completion.choices[0].message.content)
