from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from typing import List
from sqlalchemy.orm import Session
from ..core.config import settings
from ..core.database import get_db
from ..models.document import Document
import openai
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

class Message(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

class AIQueryRequest(BaseModel):
    messages: List[Message]

class AIQueryResponse(BaseModel):
    answer: str

# Retrieve all document texts from the database (up to a token limit)
def get_all_documents_context(db: Session, max_chars: int = 6000) -> str:
    docs = db.query(Document).filter(Document.extracted_text != None).all()
    context_chunks = []
    total_chars = 0
    for i, doc in enumerate(docs):
        text = doc.extracted_text or ""
        if not text.strip():
            continue
        chunk = f"Document {i+1} ({doc.original_filename}):\n{text[:1500]}"
        if total_chars + len(chunk) > max_chars:
            break
        context_chunks.append(chunk)
        total_chars += len(chunk)
    return "\n\n".join(context_chunks)

@router.post("/ai-query", response_model=AIQueryResponse)
async def ai_query(request: AIQueryRequest, db: Session = Depends(get_db)):
    """Handle AI query requests using OpenAI with all document context."""
    try:
        # Retrieve all document context (up to a safe limit)
        all_docs_context = get_all_documents_context(db, max_chars=6000)
        messages = []
        if all_docs_context:
            messages.append({
                "role": "system",
                "content": f"The following information is from all uploaded engineering documents. Use it to answer the user's question if relevant.\n\n{all_docs_context}"
            })
        # Add the rest of the chat history
        messages += [m.dict() for m in request.messages]
        client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=getattr(settings, "OPENAI_MODEL", None) or "gpt-3.5-turbo",
            messages=messages,
            max_tokens=512,
            temperature=0.7
        )
        answer = response.choices[0].message.content.strip()
        return AIQueryResponse(answer=answer)
    except Exception as e:
        logger.error(f"AI query failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="AI query failed.") 
