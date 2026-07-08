# FastAPI Endpoint Snippet
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/api/chat")
async def chat_endpoint(payload: ChatPayload):
    # Your RAG chain logic here
    async def event_generator():
        async for chunk in rag_chain.astream({"query": payload.message}):
            yield f"data: {chunk}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")