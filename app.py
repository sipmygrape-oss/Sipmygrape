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
    
    # Conversion du JSON en texte propre pour l'IA
    contexte = json.dumps(data, ensure_ascii=False)
    contexte_court = contexte[:10000] # Limite pour éviter les erreurs de taille
    
    with st.spinner('Le sommelier analyse tes goûts...'):
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "Tu es un sommelier expert. Ta tâche est d'analyser les données de la cave de l'utilisateur. "
                        "IMPORTANT : Dans tes réponses, utilise STRICTEMENT les noms tels qu'ils apparaissent dans les données. "
                        "Ne jamais inventer de nom, ne pas mélanger domaine, appellation et cépage. "
                        "Structure ta réponse ainsi :\n\n"
                        "### 🍷 Dans ma cave\n"
                        "- [Nom du vin] : [Pourquoi il correspond]\n"
                        "- [Nom du vin] : [Pourquoi il correspond]\n\n"
                        "### 🌟 Suggestion Sommelier\n"
                        "- [Nom du vin] : [Lien avec mes goûts + prix estimé]\n\n"
                        "Données de la cave : " + contexte_court
                    )
                },
                {"role": "user", "content": user_input}
            ]
        )
    
    st.write(completion.choices[0].message.content)
