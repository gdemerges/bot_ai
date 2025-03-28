import os
from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from openai import OpenAI
import openai
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Dossier contenant les documents
DOCS_DIR = "docs"

def load_documents():
    documents = []
    for filename in os.listdir(DOCS_DIR):
        path = os.path.join(DOCS_DIR, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif filename.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(path)
        elif filename.endswith(".xlsx"):
            loader = UnstructuredExcelLoader(path)
        else:
            continue
        docs = loader.load()
        documents.extend(docs)
    return documents

def process_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

def upload_to_vector_store(docs):
    print("Uploading to OpenAI vector store...")
    assistant_id = os.getenv("OPENAI_ASSISTANT_ID")
    file_ids = []

    # Charger chaque document sur OpenAI
    for doc in docs:
        with open("temp_chunk.txt", "w") as f:
            f.write(doc.page_content)

        uploaded_file = client.files.create(
            file=open("temp_chunk.txt", "rb"),
            purpose="assistants"
        )
        file_ids.append(uploaded_file.id)
        os.remove("temp_chunk.txt")

    # Obtenir les fichiers déjà attachés à l'assistant
    existing_files_response = client.beta.assistants.files.list(assistant_id=assistant_id)
    existing_file_ids = [file.id for file in existing_files_response.data]

    # Combiner les anciens et les nouveaux fichiers
    updated_file_ids = existing_file_ids + file_ids

    # Mettre à jour l'assistant avec la nouvelle liste de fichiers
    client.beta.assistants.update_files(
        assistant_id=assistant_id,
        file_ids=updated_file_ids
    )

    print("Uploaded files and attached to assistant:", file_ids)
    print("✅ Fichiers uploadés dans OpenAI et attachés à l'assistant.")

    return file_ids

if __name__ == "__main__":
    print("🔍 Chargement des documents...")
    raw_docs = load_documents()
    print(f"✔️ {len(raw_docs)} documents bruts chargés.")

    print("✂️ Découpage en chunks...")
    docs = process_documents(raw_docs)
    print(f"✔️ {len(docs)} chunks générés.")

    print("📤 Envoi dans le vector store OpenAI...")
    upload_to_vector_store(docs)
    print("✅ Fini.")
