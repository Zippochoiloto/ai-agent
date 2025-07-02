import shutil
import tempfile
from contextlib import asynccontextmanager

import requests
import whisper
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel

from app.ingest import ingest_pdf, ingest_pdf_openai, ingest_pdf_startup
from app.qa import get_qa_chain
from dotenv import load_dotenv


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize QA chain from PDF at startup
    await ingest_pdf_startup("app/data/Luong_The_Anh_FS_Developer.pdf")
    yield


app = FastAPI(lifespan=lifespan)
load_dotenv()
qa_chain = get_qa_chain()
# Global variables to keep the vector store and chain in memory
qa_chain_open_ai: RetrievalQA | None = None
vector_store: FAISS | None = None


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/upload-openai")
async def upload_pdf(file: UploadFile = File(...)):
    await ingest_pdf_openai(file)


class Question(BaseModel):
    q: str


@app.post("/ask-openai")
async def ask_question(body: Question):
    if not qa_chain_open_ai:
        return JSONResponse(status_code=400, content={"error": "No PDF has been uploaded yet."})
    response = qa_chain_open_ai.run(body.q)
    return JSONResponse(content={"answer": response})


@app.post("/ask")
def ask_question(body: Question):
    print(body.q)
    answer = qa_chain.invoke(body.q)
    print(answer)
    return {"answer": answer}


@app.post("/ask-video")
def ask_question(body: Question):
    print(body.q)
    video_qa_chain = get_qa_chain(filter_type="video")
    answer = video_qa_chain.invoke(body.q)
    print(answer)
    return {"answer": answer}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = f"app/data/{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    ingest_pdf(file_path, metadata={"type": "pdf"})
    return {"message": f"{file.filename} uploaded and ingested successfully."}


class UrlUpload(BaseModel):
    url: str


@app.post("/transcribe")
async def transcribe_video(body: UrlUpload):
    url = body.url
    # Step 1: Download video
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        return {"error": "Failed to download video."}

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                temp_file.write(chunk)
        temp_file_path = temp_file.name

    # Step 2: Transcribe
    model = whisper.load_model("base")
    result = model.transcribe(temp_file_path)

    # Step 3: Ingest to Chroma
    doc = Document(page_content=result["text"], metadata={"source": url, "type": "video"})
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    Chroma.from_documents([doc], embedding=embedding, persist_directory="app/data")

    return {"transcription": result["text"]}
