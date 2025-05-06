from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStore


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
    db.persist()
    print("✅ Ingestion complete.")
    return db


if __name__ == "__main__":
    ingest_pdf("app/data/CV-DINH-TUAN-ANH.pdf")  # add your PDF here
