import streamlit as st
import pandas as pd
from help_desk import HelpDesk

# Streamlit page configuration - must be the first Streamlit command
st.set_page_config(page_title="RAG Diagnostic", page_icon="🔍")

st.title("RAG System Diagnostic")
st.subheader("Visualize retrieved documents and similarity scores")

# Initialize the model
@st.cache_resource
def get_model():
    model = HelpDesk(new_db=False)  # Use existing database
    return model

model = get_model()

# User interface
query = st.text_input("Enter your query to test the RAG system", "Project management")

if st.button("Test"):
    with st.spinner("Retrieving documents..."):
        # Retrieve relevant documents
        docs = model.retriever.get_relevant_documents(query)
        
        # Display results
        st.subheader(f"Documents retrieved for: '{query}'")
        
        # Create DataFrame to display results
        results = []
        for i, doc in enumerate(docs):
            results.append({
                "Rank": i+1,
                "Title": doc.metadata.get("title", "Not available"),
                "Source": doc.metadata.get("source", "Not available"),
                "Content (excerpt)": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            })
        
        # Display results table
        if results:
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            
            # Display statistics
            st.subheader("Statistics")
            st.write(f"Number of documents retrieved: {len(docs)}")
            
            # Display full documents
            st.subheader("Full document content")
            for i, doc in enumerate(docs[:5]):  # Limit to 5 documents for readability
                with st.expander(f"Document {i+1}: {doc.metadata.get('title', 'Not available')}"):
                    st.write(f"**Source:** {doc.metadata.get('source', 'Not available')}")
                    st.write(f"**Content:**\n{doc.page_content}")
        else:
            st.error("No documents were retrieved for this query.")
