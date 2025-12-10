import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
# --- NEW IMPORTS FOR LOCAL OLLAMA ---
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

# --- PAGE SETUP ---
st.set_page_config(page_title="Apocalypse.ai", page_icon="☣️")
st.title("☣️ Apocalypse.ai")
st.markdown("**Status:** SYSTEM ONLINE | **Power:** BATTERY | **Connection:** LOCAL/OFFLINE")

# --- LOAD DATA & BUILD BRAIN ---
@st.cache_resource
def initialize_vector_store():
    # Check if PDF exists
    if not os.path.exists("survival.pdf"):
        st.error("CRITICAL ERROR: 'survival.pdf' not found. Please move the PDF to the ApocalypseAI folder.")
        st.stop()
        
    # 1. Load Data
    loader = PyPDFLoader("survival.pdf")
    docs = loader.load()
    
    # SAFETY CHECK: Is the PDF empty?
    if len(docs) == 0:
        st.error("❌ ERROR: The PDF file is empty or corrupt.")
        st.stop()
        
    # 2. Split Data
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    if len(splits) == 0:
        st.error("❌ ERROR: No text found in PDF! Ensure it is a text-based PDF, not images.")
        st.stop()
    
    # 3. Embed & Store (USING LOCAL OLLAMA)
    # We use 'nomic-embed-text' because it runs fast on laptops
    embeddings = OllamaEmbeddings(model="all-minilm")
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

# --- MANUAL RAG PIPELINE ---
def get_survival_advice(query, vectorstore):
    # 1. Retrieve Context
    docs = vectorstore.similarity_search(query, k=3)
    context_text = "\n\n---\n\n".join([d.page_content for d in docs])
    
    # 2. Prompt
    prompt_template = ChatPromptTemplate.from_template("""
    You are an expert survival consultant. Answer the user's question based ONLY on the following context from the US Army Survival Manual.
    If the answer is not in the context, say "I cannot find that in the manual."
    
    Context:
    {context}
    
    Question: 
    {question}
    """)
    
    # 3. Generate Answer (USING LOCAL LLAMA 3.2)
    llm = ChatOllama(model="llama3.2", temperature=0)
    chain = prompt_template | llm
    
    # Stream the response back
    response_text = chain.invoke({"context": context_text, "question": query})
    return response_text, docs

try:
    with st.spinner("Initializing Local Survival Protocols (Ollama)..."):
        vector_db = initialize_vector_store()

    st.divider()
    user_query = st.text_input("Describe your emergency:")

    if user_query:
        with st.spinner("Analyzing manual locally..."):
            # Note: Local models take a few seconds longer than cloud models
            answer_content, source_docs = get_survival_advice(user_query, vector_db)

        st.success("✅ **Tactical Recommendation:**")
        # Handle response format (Ollama sometimes returns an object, sometimes a string)
        if hasattr(answer_content, 'content'):
            st.write(answer_content.content)
        else:
            st.write(answer_content)
        
        with st.expander("View Source Manual Excerpts"):
            for i, doc in enumerate(source_docs):
                st.markdown(f"**Source {i+1} (Page {doc.metadata.get('page', '?')})**")
                st.caption(doc.page_content[:300] + "...")

except Exception as e:
    st.error(f"System Error: {e}")
