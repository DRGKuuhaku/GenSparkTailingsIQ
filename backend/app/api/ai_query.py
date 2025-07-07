from fastapi import APIRouter, HTTPException, status, Depends
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

# Set OpenAI API key from config
def get_openai_api_key():
    if settings.OPENAI_API_KEY:
        return settings.OPENAI_API_KEY
    raise RuntimeError("OpenAI API key not configured.")

@router.post("/ai-query", response_model=AIQueryResponse)
async def ai_query(request: AIQueryRequest):
    """Handle AI query requests using OpenAI."""
    openai.api_key = get_openai_api_key()
    try:
        ChatCompletion = getattr(openai, 'ChatCompletion', None)
        Completion = getattr(openai, 'Completion', None)
        if ChatCompletion is not None:
            completion = ChatCompletion.create(
                model=getattr(settings, "OPENAI_MODEL", None) or "gpt-3.5-turbo",
                messages=[{"role": "user", "content": request.query}],
                max_tokens=512,
                temperature=0.7
            )
            answer = completion.choices[0].message["content"].strip()
        elif Completion is not None:
            completion = Completion.create(
                engine=getattr(settings, "OPENAI_MODEL", None) or "text-davinci-003",
                prompt=request.query,
                max_tokens=512,
                temperature=0.7
            )
            answer = completion.choices[0].text.strip()
        else:
            raise RuntimeError("No OpenAI completion class available.")
        return AIQueryResponse(answer=answer)
    except Exception as e:
        logger.error(f"AI query failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="AI query failed.") 
