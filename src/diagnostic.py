import streamlit as st
import pandas as pd
from help_desk import HelpDesk

# Configuration de la page Streamlit - doit être la première commande Streamlit
st.set_page_config(page_title="Diagnostic RAG", page_icon="🔍")

st.title("Diagnostic du système RAG")
st.subheader("Visualisez les documents récupérés et les scores de similarité")


# Initialiser le modèle
@st.cache_resource
def get_model():
    model = HelpDesk(new_db=False)  # Utiliser la base existante
    return model


model = get_model()

# Interface utilisateur
query = st.text_input("Entrez votre requête pour tester le système RAG", "Gestion de projet")

if st.button("Tester"):
    with st.spinner("Récupération des documents..."):
        # Récupérer les documents pertinents
        docs = model.retriever.get_relevant_documents(query)

        # Afficher les résultats
        st.subheader(f"Documents récupérés pour: '{query}'")

        # Créer un DataFrame pour afficher les résultats
        results = []
        for i, doc in enumerate(docs):
            results.append(
                {
                    "Rang": i + 1,
                    "Titre": doc.metadata.get("title", "Non disponible"),
                    "Source": doc.metadata.get("source", "Non disponible"),
                    "Contenu (extrait)": (
                        doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    ),
                }
            )

        # Afficher le tableau des résultats
        if results:
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)

            # Afficher des statistiques
            st.subheader("Statistiques")
            st.write(f"Nombre de documents récupérés: {len(docs)}")

            # Afficher les documents complets
            st.subheader("Contenu complet des documents")
            for i, doc in enumerate(docs[:5]):  # Limiter à 5 documents pour la lisibilité
                with st.expander(f"Document {i + 1}: {doc.metadata.get('title', 'Non disponible')}"):
                    st.write(f"**Source:** {doc.metadata.get('source', 'Non disponible')}")
                    st.write(f"**Contenu:**\n{doc.page_content}")
        else:
            st.error("Aucun document n'a été récupéré pour cette requête.")
