from tempfile import NamedTemporaryFile

from fastapi import UploadFile
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.vectorstores import VectorStore
from langchain.chat_models import ChatOpenAI
import app.main as main


def ingest_pdf(file_path: str, persist_dir: str = "app/data", metadata: dict = None) -> VectorStore:
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    if metadata:
        for chunk in chunks:
            chunk.metadata.update(metadata)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print("✅ Ingestion complete.")
    return db


async def ingest_pdf_startup(file_path: str):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    # Embed and store with FAISS
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    main.vector_store = FAISS.from_documents(chunks, embedding=embeddings)
    # Set up retrieval QA chain
    retriever = main.vector_store.as_retriever()
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    main.qa_chain_open_ai = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


async def ingest_pdf_openai(file: UploadFile):
    # global vector_store, qa_chain_open_ai
    # Save uploaded file temporarily
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        contents = await file.read()
        temp_pdf.write(contents)
        temp_path = temp_pdf.name

    # Load and split documents
    loader = PyPDFLoader(temp_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    # Embed and store with FAISS
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    main.vector_store = FAISS.from_documents(chunks, embedding=embeddings)
    # Set up retrieval QA chain
    retriever = main.vector_store.as_retriever()
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    main.qa_chain_open_ai = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


if __name__ == "__main__":
    ingest_pdf("app/data/test.pdf")  # add your PDF here
