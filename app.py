import os
import requests
import streamlit as st
from pathlib import Path
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    DirectoryLoader,
)
import json

# Load documents
@st.cache_resource
def load_and_index_documents(folder_path):
    loaders = [
        DirectoryLoader(folder_path, glob="**/*.txt", loader_cls=TextLoader),
        DirectoryLoader(folder_path, glob="**/*.md", loader_cls=TextLoader),
        DirectoryLoader(folder_path, glob="**/*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(folder_path, glob="**/*.docx", loader_cls=UnstructuredWordDocumentLoader),
    ]
    all_docs = []
    for loader in loaders:
        all_docs.extend(loader.load())

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(all_docs)

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

# Get relevant context
def get_relevant_context(query, vector_store, k=3):
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join(doc.page_content for doc in docs)

# Call Ollama Gemma API
# def generate_with_gemma(prompt):
#     url = "http://localhost:11434/api/generate"
#     payload = {
#         "model": "gemma3:latest",
#         "prompt": prompt,
#         "stream": False
#     }
#     headers = {"Content-Type": "application/json"}
#     try:
#         response = requests.post(url, json=payload, headers=headers)
#         response.raise_for_status()
#         return response.json()["response"]
#     except requests.RequestException as e:
#         return f"Error calling Gemma API: {e}"

def generate_with_gemma_stream(prompt):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "gemma3:latest",
        "prompt": prompt,
        "stream": True
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers, stream=True)
        response.raise_for_status()

        full_response = ""
        placeholder = st.empty()

        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    token = data.get("response", "")
                    full_response += token
                    placeholder.markdown(f"### 💬 Answer:\n\n{full_response}")
                except json.JSONDecodeError as e:
                    st.error(f"Failed to parse line: {line}\n{e}")

        return full_response

    except requests.RequestException as e:
        return f"Error calling Gemma API: {e}"

# Streamlit UI
st.set_page_config(page_title="Innowation week", layout="wide")
st.title("PiCloud AI Assistant")

# Upload documents
uploaded_files = st.file_uploader("Upload documents (.txt, .pdf, .docx, .md)", accept_multiple_files=True)

# Save uploaded files to disk
doc_dir = Path("documents")
doc_dir.mkdir(exist_ok=True)

for file in uploaded_files:
    filepath = doc_dir / file.name
    if not filepath.exists():
        with open(filepath, "wb") as f:
            f.write(file.getbuffer())


# Load and index documents 
if uploaded_files:
    with st.spinner("Indexing documents..."):
        vector_store = load_and_index_documents(str(doc_dir))
    st.success("Documents indexed successfully.")

    # User query input
    question = st.text_input("Ask a question about your documents:")

    if question:
        with st.spinner("Generating answer..."):
            context = get_relevant_context(question, vector_store)
        full_prompt = f"""You are an assistant answering questions based on the following context:

{context}

Question: {question}
Answer:"""
        generate_with_gemma_stream(full_prompt)