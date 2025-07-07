from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from ..core.config import settings
import openai
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

class AIQueryRequest(BaseModel):
    query: str

class AIQueryResponse(BaseModel):
    answer: str

@router.post("/ai-query", response_model=AIQueryResponse)
async def ai_query(request: AIQueryRequest):
    """Handle AI query requests using OpenAI."""
    try:
        client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=getattr(settings, "OPENAI_MODEL", None) or "gpt-3.5-turbo",
            messages=[{"role": "user", "content": request.query}],
            max_tokens=512,
            temperature=0.7
        )
        answer = response.choices[0].message.content.strip()
        return AIQueryResponse(answer=answer)
    except Exception as e:
        logger.error(f"AI query failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="AI query failed.") 
