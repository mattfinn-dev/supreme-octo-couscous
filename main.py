from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from rag import query_chromadb

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
async def root():
    return "Yo"

@app.post("/ask_llm/")
async def ask_llm(data:dict):
    message_content = query_chromadb(prompt=data["prompt"])
    return message_content