import streamlit as st
import bcrypt
import hashlib
from pathlib import Path
import os
import json
import requests
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    DirectoryLoader,
)

# Set page config FIRST
st.set_page_config(page_title="PiCloud AI Assistant", layout="wide")

USER_FILE = "users.txt"

# ----------------------------------
# User Authentication
# ----------------------------------

def load_users():
    users = {}
    if Path(USER_FILE).exists():
        with open(USER_FILE, "r") as f:
            for line in f:
                username, hashed = line.strip().split(":")
                users[username] = hashed.encode()
    return users

def save_user(username, password):
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    with open(USER_FILE, "a") as f:
        f.write(f"{username}:{hashed.decode()}\n")

def authenticate(username, password):
    users = load_users()
    if username in users:
        return bcrypt.checkpw(password.encode(), users[username])
    return False

# ----------------------------------
# Login / Signup UI
# ----------------------------------

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    tab1, tab2 = st.tabs(["🔐 Login", "🆕 Sign Up"])

    with tab1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            if authenticate(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with tab2:
        st.subheader("Sign Up")
        new_user = st.text_input("Choose Username", key="signup_user")
        new_pass = st.text_input("Choose Password", type="password", key="signup_pass")
        if st.button("Sign Up"):
            users = load_users()
            if new_user in users:
                st.warning("Username already exists.")
            elif new_user.strip() == "" or new_pass.strip() == "":
                st.warning("Username and password cannot be empty.")
            else:
                save_user(new_user, new_pass)
                st.success("Signup successful! You can now login.")

    st.stop()

# ----------------------------------
# Main App (Only after login)
# ----------------------------------

st.title(f"Welcome, {st.session_state.username} 👋")



# ... continue with indexing, context retrieval, and generation


# Constants
DOC_DIR = Path("documents")
METADATA_FILE = DOC_DIR / ".index_metadata.json"
VECTOR_INDEX_PATH = DOC_DIR / "vector_store"

# Ensure document directory exists
DOC_DIR.mkdir(exist_ok=True)

# Uploader (optional)
uploaded_files = st.file_uploader("Upload documents (.txt, .pdf, .docx, .md) [optional]", accept_multiple_files=True)

for file in uploaded_files:
    filepath = DOC_DIR / file.name
    if not filepath.exists():
        with open(filepath, "wb") as f:
            f.write(file.getbuffer())

# Utility to calculate checksum
def calculate_file_hash(path):
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# Scan directory and return a dict of filenames + hash
def get_file_metadata(directory):
    metadata = {}
    for file in directory.glob("**/*"):
        if file.is_file() and not file.name.startswith("."):
            metadata[str(file)] = calculate_file_hash(file)
    return metadata

# Load previous index metadata if exists
def load_index_metadata():
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    return {}

# Save index metadata
def save_index_metadata(metadata):
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f)

# Load and index documents if changed
def load_and_index_documents_if_changed():
    current_meta = get_file_metadata(DOC_DIR)
    previous_meta = load_index_metadata()

    if current_meta != previous_meta or not VECTOR_INDEX_PATH.exists():
        with st.spinner("Changes detected. Re-indexing documents..."):
            loaders = [
                DirectoryLoader(str(DOC_DIR), glob="**/*.txt", loader_cls=TextLoader),
                DirectoryLoader(str(DOC_DIR), glob="**/*.md", loader_cls=TextLoader),
                DirectoryLoader(str(DOC_DIR), glob="**/*.pdf", loader_cls=PyPDFLoader),
                DirectoryLoader(str(DOC_DIR), glob="**/*.docx", loader_cls=UnstructuredWordDocumentLoader),
            ]
            all_docs = []
            for loader in loaders:
                all_docs.extend(loader.load())

            splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            docs = splitter.split_documents(all_docs)

            embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_store = FAISS.from_documents(docs, embeddings)
            vector_store.save_local(str(VECTOR_INDEX_PATH))
            save_index_metadata(current_meta)
    else:
        with st.spinner("No changes detected. Loading existing index..."):
            embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_store = FAISS.load_local(str(VECTOR_INDEX_PATH), embeddings, allow_dangerous_deserialization=True)

    return vector_store

# Query context
def get_relevant_context(query, vector_store, k=3):
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join(doc.page_content for doc in docs)

# Stream response from Ollama
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


# Load vector store
with st.spinner("Checking document index..."):
    vector_store = load_and_index_documents_if_changed()
st.success("Documents ready for querying.")

# User input
question = st.text_input("Ask a question about your documents:")

if question:
    with st.spinner("Generating answer..."):
        context = get_relevant_context(question, vector_store)
        full_prompt = f"""You are an assistant that gives very short answers and to the point answers based on the following context:

{context}

Question: {question}
Answer:"""
        generate_with_gemma_stream(full_prompt)

        with st.expander("See retrieved context"):
            st.markdown(context)
