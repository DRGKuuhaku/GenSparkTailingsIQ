"""
TailingsIQ - AI Query API
API endpoints for AI-powered query processing

This module provides REST API endpoints for natural language queries,
document search, monitoring analysis, and predictive insights.
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from ..core.database import get_db
from ..core.security import get_current_user
from ..models.user import User, UserRole
from ..services.ai_query_service import AIQueryService
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize AI query service
ai_service = AIQueryService()

# Pydantic models for request/response
class AIQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Natural language query")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context (facility_id, date_range, etc.)")
    include_sources: bool = Field(default=True, description="Include source documents in response")
    include_analysis: bool = Field(default=True, description="Include data analysis and insights")

class AIQueryResponse(BaseModel):
    response: str
    query_intent: Dict[str, Any]
    analysis: Dict[str, Any]
    data_summary: Dict[str, Any]
    recommendations: List[str]
    visualization_suggestions: List[str]
    sources: List[str]
    confidence_score: float
    processing_time: float
    timestamp: str

class QueryHistoryItem(BaseModel):
    id: int
    query: str
    response_preview: str
    timestamp: str
    confidence_score: float

class QueryHistoryResponse(BaseModel):
    queries: List[QueryHistoryItem]
    total_count: int

@router.post("/submit", response_model=AIQueryResponse)
def submit_ai_query(
    request: AIQueryRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Submit a natural language query to the AI system
    
    This endpoint processes natural language queries about:
    - Monitoring data and trends
    - Document search and analysis
    - Compliance status and requirements
    - Predictive insights and forecasting
    - Alert analysis and risk assessment
    """
    
    # Check permissions
    if not _has_ai_query_permission(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to use AI query functionality"
        )
    
    try:
        logger.info(f"Processing AI query from user {current_user.id}: {request.query[:100]}...")
        
        # Process the query
        result = ai_service.process_query(
            query=request.query,
            db=db,
            user=current_user,
            include_sources=request.include_sources,
            include_analysis=request.include_analysis
        )
        
        # Save query to history in background
        background_tasks.add_task(
            ai_service.save_query,
            current_user.id,
            request.query,
            result,
            db
        )
        
        # Format response
        response = AIQueryResponse(
            response=result.response,
            query_intent={
                "type": result.query_intent.type,
                "data_types": result.query_intent.data_types,
                "time_range": result.query_intent.time_range,
                "confidence": result.query_intent.confidence
            },
            analysis=result.analysis or {},
            data_summary=result.data_summary or {},
            recommendations=result.recommendations or [],
            visualization_suggestions=[],  # Not implemented in fallback mode
            sources=[source.get("title", "") for source in (result.sources or [])],
            confidence_score=result.confidence_score,
            processing_time=result.processing_time,
            timestamp=datetime.utcnow().isoformat()
        )
        
        logger.info(f"AI query completed for user {current_user.id} with confidence {response.confidence_score}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing AI query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process query: {str(e)}"
        )

@router.get("/history", response_model=QueryHistoryResponse)
def get_query_history(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get user's AI query history
    
    Returns a paginated list of previous queries and their responses
    """
    
    # Check permissions
    if not _has_ai_query_permission(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to access query history"
        )
    
    try:
        # Get query history
        history = ai_service.get_query_history(
            user_id=current_user.id,
            db=db,
            limit=limit
        )
        
        # Format response
        query_items = []
        for item in history:
            query_items.append(QueryHistoryItem(
                id=item.get("id", 0),
                query=item.get("query", ""),
                response_preview=item.get("response_preview", ""),
                timestamp=item.get("timestamp", ""),
                confidence_score=item.get("confidence_score", 0.0)
            ))
        
        return QueryHistoryResponse(
            queries=query_items,
            total_count=len(query_items)
        )
        
    except Exception as e:
        logger.error(f"Error getting query history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve query history"
        )

@router.post("/documents/{document_id}/index")
async def index_document_for_ai(
    document_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Index a document for AI query processing
    
    This endpoint adds a document to the AI knowledge base so it can be
    searched and referenced in future queries.
    """
    
    # Check permissions - only admin users can index documents
    if current_user.role not in [UserRole.ADMIN.value, UserRole.SUPER_ADMIN.value, UserRole.ENGINEER_OF_RECORD.value]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to index documents for AI"
        )
    
    try:
        # Add document to knowledge base in background
        background_tasks.add_task(
            ai_service.add_document_to_knowledge_base,
            document_id,
            db
        )
        
        logger.info(f"Document {document_id} queued for AI indexing by user {current_user.id}")
        
        return {
            "message": "Document queued for AI indexing",
            "document_id": document_id,
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Error indexing document for AI: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to index document for AI"
        )

@router.get("/capabilities")
async def get_ai_capabilities(
    current_user: User = Depends(get_current_user)
):
    """
    Get information about AI query capabilities
    
    Returns information about what types of queries the AI can handle
    and any current limitations.
    """
    
    # Check permissions
    if not _has_ai_query_permission(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to access AI capabilities"
        )
    
    capabilities = {
        "query_types": {
            "monitoring": {
                "description": "Analyze monitoring data, trends, and alerts",
                "examples": [
                    "Show me water level trends for the last month",
                    "What are the current critical alerts?",
                    "Analyze pore pressure data from station A"
                ]
            },
            "document": {
                "description": "Search and analyze documents and reports",
                "examples": [
                    "Find documents about stability analysis",
                    "Summarize the latest compliance report",
                    "What does the geotechnical assessment say about slope stability?"
                ]
            },
            "compliance": {
                "description": "Check compliance status and requirements",
                "examples": [
                    "What compliance requirements are due this month?",
                    "Show me the current compliance status",
                    "Which requirements need attention?"
                ]
            },
            "prediction": {
                "description": "Generate predictions and forecasts",
                "examples": [
                    "Predict water level trends for the next quarter",
                    "What are the risk factors for the next assessment?",
                    "Forecast potential stability issues"
                ]
            },
            "alert": {
                "description": "Analyze alerts and risk assessment",
                "examples": [
                    "What caused the recent critical alert?",
                    "Analyze the risk level of current alerts",
                    "What actions should be taken for these warnings?"
                ]
            }
        },
        "data_sources": [
            "Monitoring station readings",
            "Document repository",
            "Compliance assessments",
            "Historical data",
            "Regulatory requirements"
        ],
        "features": [
            "Natural language processing",
            "Semantic document search",
            "Trend analysis",
            "Risk assessment",
            "Predictive insights",
            "Automated recommendations",
            "Visualization suggestions"
        ],
        "limitations": [
            "Requires OpenAI API key for full functionality",
            "Document indexing may take time",
            "Historical data limited to available records",
            "Predictions based on available data patterns"
        ],
        "ai_status": {
            "openai_configured": bool(ai_service.llm is not None),
            "vector_store_available": bool(ai_service.vector_store is not None),
            "embeddings_available": bool(ai_service.embeddings is not None)
        }
    }
    
    return capabilities

@router.get("/health")
async def ai_health_check(
    current_user: User = Depends(get_current_user)
):
    """
    Check AI system health and status
    
    Returns the current status of AI components and their availability.
    """
    
    # Check permissions
    if not _has_ai_query_permission(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to access AI health status"
        )
    
    health_status = {
        "status": "healthy",
        "components": {
            "llm": {
                "status": "available" if ai_service.llm else "unavailable",
                "model": "OpenAI GPT-4" if ai_service.llm else "None"
            },
            "embeddings": {
                "status": "available" if ai_service.embeddings else "unavailable",
                "model": "OpenAI text-embedding-ada-002" if ai_service.embeddings else "None"
            },
            "vector_store": {
                "status": "available" if ai_service.vector_store else "unavailable",
                "type": "ChromaDB"
            }
        },
        "capabilities": {
            "natural_language_processing": bool(ai_service.llm),
            "semantic_search": bool(ai_service.embeddings and ai_service.vector_store),
            "document_analysis": bool(ai_service.llm and ai_service.embeddings),
            "monitoring_analysis": bool(ai_service.llm),
            "compliance_analysis": bool(ai_service.llm)
        }
    }
    
    # Determine overall status
    if not ai_service.llm:
        health_status["status"] = "degraded"
        health_status["message"] = "AI responses limited - OpenAI not configured"
    elif not ai_service.embeddings or not ai_service.vector_store:
        health_status["status"] = "degraded"
        health_status["message"] = "Document search limited - vector store not available"
    
    return health_status

def _has_ai_query_permission(user: User) -> bool:
    """Check if user has permission to use AI query functionality"""
    allowed_roles = [
        UserRole.SUPER_ADMIN.value,
        UserRole.ADMIN.value,
        UserRole.ENGINEER_OF_RECORD.value,
        UserRole.TSF_OPERATOR.value,
        UserRole.REGULATOR.value,
        UserRole.MANAGEMENT.value,
        UserRole.CONSULTANT.value
    ]
    
    return user.role in allowed_roles 
