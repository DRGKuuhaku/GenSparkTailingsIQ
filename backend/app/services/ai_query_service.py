import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from sqlalchemy.orm import Session
from sqlalchemy import select

from ..models.monitoring import MonitoringReading, MonitoringAlert
from ..models.document import Document
from ..models.compliance import ComplianceAssessment
from ..models.user import User
import openai
import os

logger = logging.getLogger(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")

@dataclass
class QueryIntent:
    type: str
    data_types: List[str]
    time_range: Optional[str] = None
    confidence: float = 0.0

@dataclass
class QueryResponse:
    response: str
    query_intent: QueryIntent
    confidence_score: float
    processing_time: float
    analysis: Optional[Dict[str, List[str]]] = None
    recommendations: Optional[List[str]] = None
    data_summary: Optional[Dict[str, int]] = None
    sources: Optional[List[Dict[str, Any]]] = None

class AIQueryService:
    def __init__(self):
        logger.info("AI Query Service initialized (fallback mode)")
    
    def _analyze_query_intent(self, query: str) -> QueryIntent:
        """Analyze the intent of a user query"""
        try:
            query_lower = query.lower()
            
            # Define intent patterns
            intent_patterns = {
                "alert": ["alert", "alarm", "critical", "warning", "issue", "problem"],
                "monitoring": ["monitor", "reading", "sensor", "data", "trend", "level", "pressure"],
                "document": ["document", "report", "file", "pdf", "analysis", "study"],
                "compliance": ["compliance", "regulation", "requirement", "standard", "audit"],
                "prediction": ["predict", "forecast", "future", "trend", "next", "upcoming"],
                "analysis": ["analyze", "analysis", "insight", "pattern", "correlation"]
            }
            
            # Determine primary intent
            primary_intent = "general"
            max_matches = 0
            
            for intent, keywords in intent_patterns.items():
                matches = sum(1 for keyword in keywords if keyword in query_lower)
                if matches > max_matches:
                    max_matches = matches
                    primary_intent = intent
            
            # Determine data types
            data_types = []
            if any(word in query_lower for word in ["water", "level", "sensor", "monitor"]):
                data_types.append("monitoring")
            if any(word in query_lower for word in ["document", "report", "file", "pdf"]):
                data_types.append("documents")
            if any(word in query_lower for word in ["compliance", "regulation", "requirement"]):
                data_types.append("compliance")
            
            # Determine time range
            time_range = None
            if any(word in query_lower for word in ["today", "current", "now"]):
                time_range = "current"
            elif any(word in query_lower for word in ["week", "7 days"]):
                time_range = "last_week"
            elif any(word in query_lower for word in ["month", "30 days"]):
                time_range = "last_month"
            elif any(word in query_lower for word in ["quarter", "3 months"]):
                time_range = "last_quarter"
            elif any(word in query_lower for word in ["year", "annual"]):
                time_range = "last_year"
            
            confidence = min(0.9, 0.3 + (max_matches * 0.2))
            
            return QueryIntent(
                type=primary_intent,
                data_types=data_types,
                time_range=time_range,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error analyzing query intent: {e}")
            return QueryIntent(type="general", data_types=[], confidence=0.1)
    
    def _gather_monitoring_data(self, db: Session, intent: QueryIntent) -> List[Dict[str, Any]]:
        """Gather relevant monitoring data"""
        try:
            data = []
            
            # Get recent monitoring readings
            query = select(MonitoringReading).order_by(MonitoringReading.timestamp.desc()).limit(50)
            result = db.execute(query)
            readings = result.scalars().all()
            
            for reading in readings:
                data.append({
                    "type": "monitoring",
                    "timestamp": reading.timestamp.isoformat(),
                    "station": reading.station_id,
                    "value": reading.value,
                    "unit": reading.unit,
                    "quality_code": reading.quality_code,
                    "alert_level": reading.alert_level
                })
            
            # Get active alerts
            alert_query = select(MonitoringAlert).where(MonitoringAlert.is_active == True)
            alert_result = db.execute(alert_query)
            alerts = alert_result.scalars().all()
            
            for alert in alerts:
                data.append({
                    "type": "alert",
                    "timestamp": alert.created_at.isoformat(),
                    "station": alert.station_id,
                    "alert_level": alert.alert_level,
                    "message": alert.message,
                    "alert_type": alert.alert_type,
                    "is_active": alert.is_active
                })
            
            return data
            
        except Exception as e:
            logger.error(f"Error gathering monitoring data: {e}")
            return []
    
    def _gather_document_data(self, db: Session, intent: QueryIntent) -> List[Dict[str, Any]]:
        """Gather relevant document data"""
        try:
            data = []
            
            # Get recent documents
            query = select(Document).order_by(Document.uploaded_at.desc()).limit(20)
            result = db.execute(query)
            documents = result.scalars().all()
            
            for doc in documents:
                data.append({
                    "type": "document",
                    "id": doc.id,
                    "title": doc.title,
                    "category": doc.category,
                    "uploaded_at": doc.uploaded_at.isoformat(),
                    "file_size": doc.file_size,
                    "description": doc.description or ""
                })
            
            return data
            
        except Exception as e:
            logger.error(f"Error gathering document data: {e}")
            return []
    
    def _gather_compliance_data(self, db: Session, intent: QueryIntent) -> List[Dict[str, Any]]:
        """Gather relevant compliance data"""
        try:
            data = []
            
            # Get recent compliance assessments
            query = select(ComplianceAssessment).order_by(ComplianceAssessment.assessment_date.desc()).limit(20)
            result = db.execute(query)
            assessments = result.scalars().all()
            
            for assessment in assessments:
                data.append({
                    "type": "compliance",
                    "id": assessment.id,
                    "regulation": assessment.regulation,
                    "assessment_date": assessment.assessment_date.isoformat(),
                    "status": assessment.status,
                    "score": assessment.compliance_score,
                    "notes": assessment.notes or ""
                })
            
            return data
            
        except Exception as e:
            logger.error(f"Error gathering compliance data: {e}")
            return []
    
    def _generate_response(self, query: str, context_data: Dict[str, List[Dict[str, Any]]], intent: QueryIntent) -> str:
        """Generate response based on query and available data"""
        query_lower = query.lower()
        
        # Simple keyword-based responses
        if any(word in query_lower for word in ["alert", "alarm", "critical"]):
            alerts = context_data.get("monitoring", [])
            alert_count = len([a for a in alerts if a.get("type") == "alert"])
            
            if alert_count > 0:
                return f"I found {alert_count} active alerts in the system. Please check the monitoring dashboard for detailed information about these alerts and their severity levels."
            else:
                return "No active alerts are currently detected in the system. All monitoring parameters appear to be within normal ranges."
        
        elif any(word in query_lower for word in ["water", "level", "monitor"]):
            readings = context_data.get("monitoring", [])
            reading_count = len([r for r in readings if r.get("type") == "monitoring"])
            
            if reading_count > 0:
                return f"I found {reading_count} recent monitoring readings. The latest water level and sensor data can be viewed in the monitoring dashboard. Consider checking for any trends or anomalies in the data."
            else:
                return "No recent monitoring data is available. Please check the monitoring system status and ensure sensors are functioning properly."
        
        elif any(word in query_lower for word in ["document", "report", "file"]):
            docs = context_data.get("documents", [])
            doc_count = len(docs)
            
            if doc_count > 0:
                return f"I found {doc_count} documents in the system. You can search and access these documents through the documents section. Recent uploads include various reports and analysis documents."
            else:
                return "No documents are currently available in the system. Consider uploading relevant reports, analysis documents, or compliance records."
        
        elif any(word in query_lower for word in ["compliance", "regulation", "requirement"]):
            assessments = context_data.get("compliance", [])
            assessment_count = len(assessments)
            
            if assessment_count > 0:
                return f"I found {assessment_count} compliance assessments in the system. These assessments track regulatory compliance and can be reviewed in the compliance section."
            else:
                return "No compliance assessments are currently available. Consider conducting regular compliance reviews and uploading assessment results to the system."
        
        else:
            return "I understand your query about the TSF system. To provide more specific information, I would need access to relevant data. Please check the monitoring dashboard, documents section, or compliance records for detailed information. If you have specific questions about alerts, water levels, documents, or compliance, I can help guide you to the appropriate data sources."
    
    def _analyze_data_trends(self, data: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Analyze data for trends and patterns"""
        analysis = {
            "trends": [],
            "anomalies": [],
            "risks": []
        }
        
        try:
            # Simple trend analysis
            if data:
                # Check for alerts
                alerts = [item for item in data if item.get("type") == "alert"]
                if alerts:
                    analysis["risks"].append(f"Found {len(alerts)} active alerts requiring attention")
                
                # Check for recent activity
                recent_items = [item for item in data if "timestamp" in item]
                if recent_items:
                    analysis["trends"].append(f"Recent activity detected with {len(recent_items)} data points")
                
                # Check for compliance issues
                compliance_items = [item for item in data if item.get("type") == "compliance"]
                if compliance_items:
                    low_scores = [item for item in compliance_items if item.get("score", 100) < 80]
                    if low_scores:
                        analysis["risks"].append(f"Found {len(low_scores)} compliance assessments with scores below 80%")
            
        except Exception as e:
            logger.error(f"Error analyzing data trends: {e}")
        
        return analysis
    
    def _generate_recommendations(self, intent: QueryIntent, analysis: Dict[str, List[str]]) -> List[str]:
        """Generate recommendations based on intent and analysis"""
        recommendations = []
        
        try:
            if intent.type == "alert":
                recommendations.extend([
                    "Review all active alerts immediately",
                    "Check sensor calibration and maintenance schedules",
                    "Update emergency response procedures if needed"
                ])
            
            elif intent.type == "monitoring":
                recommendations.extend([
                    "Regularly review monitoring data for trends",
                    "Set up automated alerts for critical parameters",
                    "Schedule sensor maintenance and calibration"
                ])
            
            elif intent.type == "compliance":
                recommendations.extend([
                    "Schedule regular compliance reviews",
                    "Update compliance documentation as needed",
                    "Train staff on regulatory requirements"
                ])
            
            elif intent.type == "document":
                recommendations.extend([
                    "Organize documents by category and date",
                    "Regularly update technical documentation",
                    "Ensure all reports are properly archived"
                ])
            
            # Add general recommendations
            if analysis.get("risks"):
                recommendations.append("Address identified risks promptly")
            
            if not recommendations:
                recommendations.append("Continue monitoring system performance and data quality")
                
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations = ["Monitor system performance and data quality"]
        
        return recommendations
    
    def process_query(
        self, 
        query: str, 
        db: Session, 
        user: User,
        include_sources: bool = True,
        include_analysis: bool = True
    ) -> QueryResponse:
        """Process a user query and return AI response"""
        start_time = time.time()
        
        try:
            # Call OpenAI API
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # or "gpt-4" if you have access
                messages=[
                    {"role": "system", "content": "You are an AI assistant for tailings facility management."},
                    {"role": "user", "content": query}
                ]
            )
            ai_response = completion.choices[0].message.content

            processing_time = time.time() - start_time

            # You can still analyze intent, etc., as before
            intent = self._analyze_query_intent(query)

            return QueryResponse(
                response=ai_response,
                query_intent=intent,
                confidence_score=0.9,  # You can set this based on your logic
                processing_time=processing_time,
                analysis=None,
                recommendations=None,
                data_summary=None,
                sources=None
            )
        except Exception as e:
            logger.error(f"Error processing query with OpenAI: {e}")
            processing_time = time.time() - start_time
            return QueryResponse(
                response="Sorry, there was an error processing your query with OpenAI.",
                query_intent=self._analyze_query_intent(query),
                confidence_score=0.0,
                processing_time=processing_time
            )
    
    def _prepare_sources(self, context_data: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Prepare source information for the response"""
        sources = []
        
        try:
            for data_type, items in context_data.items():
                for item in items[:5]:  # Limit to 5 items per type
                    source = {
                        "type": data_type,
                        "id": item.get("id"),
                        "title": item.get("title") or item.get("station") or f"{data_type.title()} Data",
                        "timestamp": item.get("timestamp") or item.get("uploaded_at") or item.get("assessment_date"),
                        "description": item.get("description") or item.get("message") or ""
                    }
                    sources.append(source)
                    
        except Exception as e:
            logger.error(f"Error preparing sources: {e}")
        
        return sources
    
    def index_document_for_ai(self, document_content: str, document_id: str, metadata: Dict[str, Any]) -> bool:
        """Index a document for AI search (placeholder for now)"""
        logger.info(f"Document indexing requested for document {document_id} (not implemented in fallback mode)")
        return True
    
    def get_ai_capabilities(self) -> Dict[str, Any]:
        """Get AI system capabilities and status"""
        return {
            "ai_status": {
                "langchain": False,
                "openai": False,
                "chromadb": False,
                "vector_store": False,
                "llm": False,
                "embeddings": False
            },
            "query_types": {
                "monitoring": {
                    "description": "Query monitoring data, sensor readings, and alerts",
                    "examples": [
                        "What are the current monitoring alerts?",
                        "Show me water level trends for the last month",
                        "Are there any critical sensor readings?"
                    ]
                },
                "documents": {
                    "description": "Search and analyze documents and reports",
                    "examples": [
                        "Find documents about stability analysis",
                        "Show me recent technical reports",
                        "Search for compliance documentation"
                    ]
                },
                "compliance": {
                    "description": "Query compliance assessments and regulatory requirements",
                    "examples": [
                        "What compliance requirements are due this month?",
                        "Show me recent compliance assessments",
                        "Are we meeting regulatory standards?"
                    ]
                },
                "analysis": {
                    "description": "Get AI-powered analysis and insights",
                    "examples": [
                        "Analyze the risk factors for our TSF",
                        "What trends do you see in the monitoring data?",
                        "Provide recommendations for improving safety"
                    ]
                },
                "prediction": {
                    "description": "Get predictions and forecasts based on data",
                    "examples": [
                        "Predict water level trends for the next quarter",
                        "Forecast potential issues based on current data",
                        "What might happen if current trends continue?"
                    ]
                }
            },
            "features": {
                "natural_language_processing": False,
                "document_search": False,
                "data_analysis": True,
                "trend_analysis": True,
                "recommendations": True,
                "multi_source_integration": True
            }
        }
    
    def get_ai_health(self) -> Dict[str, Any]:
        """Get AI system health status"""
        return {
            "status": "limited",
            "components": {
                "langchain": False,
                "openai": False,
                "chromadb": False,
                "vector_store": False
            },
            "capabilities": {
                "ai_responses": False,
                "document_search": False,
                "embeddings": False
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_query_history(self, user_id: int, db: Session, limit: int = 20) -> List[Dict[str, Any]]:
        """Get user's query history (placeholder implementation)"""
        try:
            # For now, return empty list since we don't have a query history table
            # In a full implementation, this would query a database table
            logger.info(f"Query history requested for user {user_id} (not implemented in fallback mode)")
            return []
        except Exception as e:
            logger.error(f"Error getting query history: {e}")
            return []
    
    def save_query(self, user_id: int, query: str, result: QueryResponse, db: Session) -> bool:
        """Save query and result to history (placeholder implementation)"""
        try:
            # For now, just log the query since we don't have a query history table
            # In a full implementation, this would save to a database table
            logger.info(f"Query saved for user {user_id}: {query[:100]}... (not implemented in fallback mode)")
            return True
        except Exception as e:
            logger.error(f"Error saving query: {e}")
            return False
    
    def add_document_to_knowledge_base(self, document_id: int, db: Session) -> bool:
        """Add document to AI knowledge base (placeholder implementation)"""
        try:
            # For now, just log the request since we don't have full AI integration
            # In a full implementation, this would process and index the document
            logger.info(f"Document {document_id} added to knowledge base (not implemented in fallback mode)")
            return True
        except Exception as e:
            logger.error(f"Error adding document to knowledge base: {e}")
            return False 
