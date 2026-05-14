from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from app.main import ingest_urls, chat as rag_chat, clear_data # Added clear_data
import uvicorn

app = FastAPI(title="Advanced RAG Chatbot API")

# Data models for requests
class IngestRequest(BaseModel):
    urls: List[str]

class ChatRequest(BaseModel):
    session_id: str
    query: str

@app.post("/ingest")
async def handle_ingestion(request: IngestRequest):
    try:
        ingest_urls(request.urls)
        return {"message": "Ingestion successful", "urls": request.urls}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def handle_chat(request: ChatRequest):
    try:
        # Note: This returns the generator/response from app.main.chat
        response = rag_chat(request.query, session_id=request.session_id)
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear")
async def handle_clear():
    """
    Endpoint to trigger a full system wipe.
    Deletes physical files and re-initializes global variables.
    """
    try:
        clear_data()
        return {"message": "Knowledge base and memory cleared successfully."}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)