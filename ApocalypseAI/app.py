import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

# --- PAGE SETUP ---
st.set_page_config(page_title="Apocalypse.ai", page_icon="☣️")
st.title("☣️ Apocalypse.ai")
st.markdown("**Status:** SYSTEM ONLINE | **Power:** BATTERY | **Connection:** LOCAL/OFFLINE")

# --- LOAD DATA & BUILD BRAIN ---
@st.cache_resource
def initialize_vector_store():
    # 1. SCAN FOLDER FOR PDFs
    # This looks for ANY file ending in .pdf in the current folder
    pdf_files = [f for f in os.listdir(".") if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        st.error("CRITICAL ERROR: No PDF files found in this folder. Please drag and drop your manuals here.")
        st.stop()
        
    docs = []
    status_text = st.empty() # Placeholder for progress updates
    
    # 2. LOAD ALL PDFs
    for pdf in pdf_files:
        status_text.text(f"Loading manual: {pdf}...")
        try:
            loader = PyPDFLoader(pdf)
            docs.extend(loader.load())
        except Exception as e:
            st.warning(f"⚠️ Could not read {pdf}: {e}")
            
    status_text.text(f"Processing {len(docs)} pages from {len(pdf_files)} manuals...")
    
    # SAFETY CHECK: Is the list empty?
    if len(docs) == 0:
        st.error("❌ ERROR: All PDFs were empty or unreadable.")
        st.stop()
        
    # 3. SPLIT DATA
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    if len(splits) == 0:
        st.error("❌ ERROR: No text found! Ensure your PDFs are text-based, not images.")
        st.stop()
    
    # 4. EMBED & STORE
    status_text.text("Embedding knowledge (this may take a minute)...")
    embeddings = OllamaEmbeddings(model="all-minilm")
    
    try:
        vectorstore = FAISS.from_documents(splits, embeddings)
    except Exception as e:
        st.error(f"❌ OLLAMA ERROR: The AI 'Brain' is not responding. Make sure Ollama is running! Error: {e}")
        st.stop()
        
    status_text.empty() # Clear the status message
    return vectorstore

# --- MANUAL RAG PIPELINE ---
def get_survival_advice(query, vectorstore):
    # Retrieve top 4 chunks (increased from 3 since we have more data now)
    docs = vectorstore.similarity_search(query, k=4)
    context_text = "\n\n---\n\n".join([d.page_content for d in docs])
    
    prompt_template = ChatPromptTemplate.from_template("""
    You are an expert survival consultant. Answer the user's question based ONLY on the following context from the provided manuals.
    If the answer is not in the context, say "I cannot find that in the manuals."
    
    Context:
    {context}
    
    Question: 
    {question}
    """)
    
    llm = ChatOllama(model="llama3.2", temperature=0)
    chain = prompt_template | llm
    
    response_text = chain.invoke({"context": context_text, "question": query})
    return response_text, docs

try:
    with st.spinner("Initializing Survival Library..."):
        vector_db = initialize_vector_store()

    st.divider()
    user_query = st.text_input("Describe your emergency:")

    if user_query:
        with st.spinner("Analyzing all manuals..."):
            answer_content, source_docs = get_survival_advice(user_query, vector_db)

        st.success("✅ **Tactical Recommendation:**")
        
        if hasattr(answer_content, 'content'):
            st.write(answer_content.content)
        else:
            st.write(answer_content)
        
        with st.expander("View Source Manual Excerpts"):
            for i, doc in enumerate(source_docs):
                # Show which specific file the info came from
                source_name = doc.metadata.get('source', 'Unknown Manual')
                st.markdown(f"**Source {i+1}:** *{source_name}* (Page {doc.metadata.get('page', '?')})")
                st.caption(doc.page_content[:300] + "...")

except Exception as e:
    st.error(f"System Error: {e}")