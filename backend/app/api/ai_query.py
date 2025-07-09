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

# Simple keyword search for relevant document text
def keyword_search_documents(db: Session, query: str, top_k: int = 1) -> List[str]:
    # Search for documents containing any keyword from the query
    keywords = [w.lower() for w in query.split() if len(w) > 2]
    results = []
    for doc in db.query(Document).filter(Document.extracted_text != None):
        text = doc.extracted_text or ""
        if any(kw in text.lower() for kw in keywords):
            results.append(text)
    # Return the top_k longest matches (as a simple heuristic)
    results = sorted(results, key=len, reverse=True)[:top_k]
    return results

@router.post("/ai-query", response_model=AIQueryResponse)
async def ai_query(request: AIQueryRequest, db: Session = Depends(get_db)):
    """Handle AI query requests using OpenAI with RAG (keyword search)."""
    try:
        # Get the latest user question
        user_question = next((m.content for m in reversed(request.messages) if m.role == 'user'), None)
        # Retrieve relevant document text
        doc_contexts = keyword_search_documents(db, user_question or "", top_k=1)
        # Prepend document context as a system message if found
        messages = []
        if doc_contexts:
            messages.append({
                "role": "system",
                "content": f"The following information is from uploaded engineering documents. Use it to answer the user's question if relevant.\n\n{doc_contexts[0][:2000]}"
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
