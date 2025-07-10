from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from typing import List
from sqlalchemy.orm import Session
from ..core.config import settings
from ..core.database import get_db
from ..models.document import Document
from ..models.synthetic_data_models import DatasetRow
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

# Helper to get relevant dataset rows for a query
def get_relevant_dataset_rows(db: Session, query: str, max_rows: int = 20) -> List[dict]:
    # Simple keyword search: include rows where any value matches a keyword in the query
    keywords = [w.lower() for w in query.split() if len(w) > 2]
    rows = db.query(DatasetRow).all()
    relevant = []
    for row in rows:
        row_dict = row.row_data if row.row_data is not None else {}
        if any(kw in str(v).lower() for kw in keywords for v in row_dict.values()):
            relevant.append(row_dict)
        if len(relevant) >= max_rows:
            break
    # If no relevant rows found, fallback to first N rows
    fallback_rows = [row.row_data if isinstance(row.row_data, dict) else {} for row in rows[:max_rows]]
    if not relevant:
        relevant = fallback_rows
    else:
        # Filter out any non-dict values just in case
        relevant = [r for r in relevant if isinstance(r, dict)]
    return relevant

# Helper to format dataset rows as markdown table
def format_rows_as_markdown(rows: List[dict]) -> str:
    if not rows:
        return "No relevant data found."
    headers = list(rows[0].keys())
    table = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        table.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(table)

@router.post("/ai-query", response_model=AIQueryResponse)
async def ai_query(request: AIQueryRequest, db: Session = Depends(get_db)):
    """Handle AI query requests using OpenAI with all document and dataset context."""
    try:
        # Retrieve all document context (up to a safe limit)
        from ..models.document import Document  # Avoid circular import at top
        def get_all_documents_context(db: Session, max_chars: int = 6000, chunk_size: int = 500) -> str:
            docs = db.query(Document).filter(Document.extracted_text != None).order_by(Document.uploaded_at.desc()).all()
            context_chunks = []
            total_chars = 0
            for i, doc in enumerate(docs):
                text = str(doc.extracted_text or '')
                if not text.strip():
                    continue
                chunk = f"Document {i+1} ({doc.original_filename}):\n{text[:chunk_size]}"
                if total_chars + len(chunk) > max_chars:
                    break
                context_chunks.append(chunk)
                total_chars += len(chunk)
            return "\n\n".join(context_chunks)

        all_docs_context = get_all_documents_context(db, max_chars=4000, chunk_size=400)
        # Get latest user question
        user_question = next((m.content for m in reversed(request.messages) if m.role == 'user'), None) or ""
        # Retrieve relevant dataset rows
        dataset_rows = get_relevant_dataset_rows(db, user_question, max_rows=15)
        dataset_context = format_rows_as_markdown(dataset_rows)
        messages = []
        if all_docs_context:
            messages.append({
                "role": "system",
                "content": f"The following information is from all uploaded engineering documents. Use it to answer the user's question if relevant.\n\n{all_docs_context}"
            })
        if dataset_context:
            messages.append({
                "role": "system",
                "content": f"The following table contains relevant data from uploaded datasets. Use it to answer the user's question if relevant.\n\n{dataset_context}"
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
