from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Import the asynchronous generator from your backend file
from bot_backend import ask_personal_bot_stream

app = FastAPI(title="Personal Portfolio RAG Bot API")

origins = [
    "http://localhost:5174",  # Vite default local port
    "http://localhost:3000",
    "https://pradeepkumarsingha.netlify.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatPayload(BaseModel):
    message: str
    session_id: str = "portfolio_user"  # Allows unique sessions per visitor

@app.post("/api/chat")
async def chat_endpoint(payload: ChatPayload):
    
    async def event_generator():
        try:
            # Consume chunks from our async RAG function
            async for text_chunk in ask_personal_bot_stream(payload.message, payload.session_id):
                if text_chunk:
                    # Clean Server-Sent Events format for UI consumption
                    yield f"data: {text_chunk}\n\n"
        except Exception as e:
            yield f"data: [Internal Error: {str(e)}]\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/")
def read_root():
    return {"status": "online", "message": "Welcome to the Portfolio RAG Chatbot API"}