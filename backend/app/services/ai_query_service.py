"""
TailingsIQ - AI Query Service
Advanced AI-powered query processing for TSF management

This service provides intelligent query processing, document search,
monitoring data analysis, and predictive insights for tailings management.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import re

# AI/ML imports
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Data processing
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Database
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

from ..core.config import settings
from ..models.document import Document as DocumentModel, DocumentChunk
from ..models.monitoring import MonitoringReading, MonitoringStation, MonitoringAlert
from ..models.compliance import ComplianceAssessment, ComplianceRequirement
from ..models.user import User

logger = logging.getLogger(__name__)

class AIQueryService:
    """Advanced AI query processing service for TailingsIQ"""
    
    def __init__(self):
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self._initialize_ai_components()
    
    def _initialize_ai_components(self):
        """Initialize AI components based on configuration"""
        try:
            # Initialize OpenAI components
            if settings.OPENAI_API_KEY:
                self.llm = ChatOpenAI(
                    model_name=settings.OPENAI_MODEL,
                    temperature=settings.OPENAI_TEMPERATURE,
                    max_tokens=settings.OPENAI_MAX_TOKENS,
                    openai_api_key=settings.OPENAI_API_KEY
                )
                self.embeddings = OpenAIEmbeddings(
                    model=settings.OPENAI_EMBEDDING_MODEL,
                    openai_api_key=settings.OPENAI_API_KEY
                )
                logger.info("OpenAI components initialized successfully")
            else:
                # Fallback to local models
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=settings.HUGGINGFACE_MODEL
                )
                logger.warning("OpenAI not configured, using local embeddings only")
            
            # Initialize vector store
            self._initialize_vector_store()
            
        except Exception as e:
            logger.error(f"Error initializing AI components: {e}")
            raise
    
    def _initialize_vector_store(self):
        """Initialize ChromaDB vector store"""
        try:
            chroma_config = settings.get_chroma_config()
            self.vector_store = Chroma(
                persist_directory=chroma_config["persist_directory"],
                embedding_function=self.embeddings,
                collection_name=chroma_config["collection_name"]
            )
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    async def process_query(self, 
                          query: str, 
                          user_id: int,
                          db: Session,
                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a natural language query and return intelligent response
        
        Args:
            query: Natural language query from user
            user_id: ID of the user making the query
            db: Database session
            context: Additional context (facility_id, date_range, etc.)
        
        Returns:
            Dictionary containing response, sources, and metadata
        """
        try:
            # Analyze query intent
            intent = self._analyze_query_intent(query)
            
            # Get relevant data based on intent
            relevant_data = await self._gather_relevant_data(
                query, intent, user_id, db, context
            )
            
            # Generate response based on intent and data
            response = await self._generate_response(
                query, intent, relevant_data, user_id, db
            )
            
            # Add metadata and sources
            response.update({
                "query_intent": intent,
                "sources": relevant_data.get("sources", []),
                "confidence_score": relevant_data.get("confidence", 0.8),
                "processing_time": relevant_data.get("processing_time", 0),
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "error": "Failed to process query",
                "message": str(e),
                "query_intent": "error",
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze the intent of the user query"""
        query_lower = query.lower()
        
        intent = {
            "type": "general",
            "entities": [],
            "data_types": [],
            "time_range": None,
            "facility_specific": False
        }
        
        # Detect query types
        if any(word in query_lower for word in ["monitor", "reading", "sensor", "data"]):
            intent["type"] = "monitoring"
            intent["data_types"].append("monitoring")
        
        if any(word in query_lower for word in ["document", "report", "file", "upload"]):
            intent["type"] = "document"
            intent["data_types"].append("document")
        
        if any(word in query_lower for word in ["compliance", "regulation", "requirement"]):
            intent["type"] = "compliance"
            intent["data_types"].append("compliance")
        
        if any(word in query_lower for word in ["predict", "forecast", "trend", "future"]):
            intent["type"] = "prediction"
            intent["data_types"].extend(["monitoring", "analysis"])
        
        if any(word in query_lower for word in ["alert", "warning", "critical", "issue"]):
            intent["type"] = "alert"
            intent["data_types"].append("monitoring")
        
        # Detect time references
        time_patterns = {
            "last_24h": r"last 24 hours?|yesterday|today",
            "last_week": r"last week|past 7 days",
            "last_month": r"last month|past 30 days",
            "last_year": r"last year|past 12 months"
        }
        
        for time_key, pattern in time_patterns.items():
            if re.search(pattern, query_lower):
                intent["time_range"] = time_key
                break
        
        # Detect facility references
        if re.search(r"facility|tsf|dam|site", query_lower):
            intent["facility_specific"] = True
        
        return intent
    
    async def _gather_relevant_data(self, 
                                  query: str, 
                                  intent: Dict[str, Any],
                                  user_id: int,
                                  db: Session,
                                  context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Gather relevant data based on query intent"""
        relevant_data = {
            "documents": [],
            "monitoring_data": [],
            "compliance_data": [],
            "sources": [],
            "confidence": 0.8,
            "processing_time": 0
        }
        
        start_time = datetime.utcnow()
        
        try:
            # Get user permissions
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise ValueError("User not found")
            
            # Gather document data if needed
            if "document" in intent["data_types"]:
                relevant_data["documents"] = await self._search_documents(
                    query, user, db, context
                )
            
            # Gather monitoring data if needed
            if "monitoring" in intent["data_types"]:
                relevant_data["monitoring_data"] = await self._get_monitoring_data(
                    query, intent, user, db, context
                )
            
            # Gather compliance data if needed
            if "compliance" in intent["data_types"]:
                relevant_data["compliance_data"] = await self._get_compliance_data(
                    query, intent, user, db, context
                )
            
            # Calculate confidence based on data availability
            total_sources = len(relevant_data["documents"]) + len(relevant_data["monitoring_data"]) + len(relevant_data["compliance_data"])
            relevant_data["confidence"] = min(0.95, 0.5 + (total_sources * 0.1))
            
            relevant_data["processing_time"] = (datetime.utcnow() - start_time).total_seconds()
            
        except Exception as e:
            logger.error(f"Error gathering relevant data: {e}")
            relevant_data["confidence"] = 0.3
        
        return relevant_data
    
    async def _search_documents(self, 
                              query: str, 
                              user: User, 
                              db: Session,
                              context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search documents using semantic search"""
        try:
            # Get document chunks from database
            chunks_query = db.query(DocumentChunk).join(DocumentModel)
            
            # Apply user permissions
            if user.role not in ["super_admin", "admin", "engineer_of_record"]:
                # Filter by user's facility access
                chunks_query = chunks_query.filter(DocumentModel.facility_id == user.facility_id)
            
            # Apply context filters
            if context and context.get("facility_id"):
                chunks_query = chunks_query.filter(DocumentModel.facility_id == context["facility_id"])
            
            if context and context.get("document_type"):
                chunks_query = chunks_query.filter(DocumentModel.document_type == context["document_type"])
            
            chunks = chunks_query.limit(50).all()
            
            if not chunks:
                return []
            
            # Create documents for vector search
            documents = []
            for chunk in chunks:
                doc = Document(
                    page_content=chunk.chunk_text,
                    metadata={
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                        "title": chunk.document.title,
                        "document_type": chunk.document.document_type,
                        "facility_id": chunk.document.facility_id
                    }
                )
                documents.append(doc)
            
            # Perform semantic search
            if self.vector_store and documents:
                # Add documents to vector store temporarily
                temp_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
                
                # Search
                results = temp_store.similarity_search_with_relevance_scores(
                    query, k=10
                )
                
                # Format results
                search_results = []
                for doc, score in results:
                    if score > 0.7:  # Relevance threshold
                        search_results.append({
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "relevance_score": score,
                            "source": f"Document: {doc.metadata.get('title', 'Unknown')}"
                        })
                
                return search_results
            
            return []
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    async def _get_monitoring_data(self, 
                                 query: str, 
                                 intent: Dict[str, Any],
                                 user: User, 
                                 db: Session,
                                 context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get relevant monitoring data"""
        try:
            # Build query
            readings_query = db.query(MonitoringReading).join(MonitoringStation)
            
            # Apply time filter
            if intent.get("time_range"):
                time_filters = {
                    "last_24h": datetime.utcnow() - timedelta(days=1),
                    "last_week": datetime.utcnow() - timedelta(days=7),
                    "last_month": datetime.utcnow() - timedelta(days=30),
                    "last_year": datetime.utcnow() - timedelta(days=365)
                }
                if intent["time_range"] in time_filters:
                    readings_query = readings_query.filter(
                        MonitoringReading.timestamp >= time_filters[intent["time_range"]]
                    )
            
            # Apply facility filter
            if context and context.get("facility_id"):
                readings_query = readings_query.filter(
                    MonitoringStation.facility_id == context["facility_id"]
                )
            
            # Get recent readings
            readings = readings_query.order_by(
                MonitoringReading.timestamp.desc()
            ).limit(100).all()
            
            # Format data
            monitoring_data = []
            for reading in readings:
                monitoring_data.append({
                    "station_id": reading.station_id,
                    "timestamp": reading.timestamp.isoformat(),
                    "value": reading.value,
                    "unit": reading.unit,
                    "alert_level": reading.alert_level,
                    "is_anomaly": reading.is_anomaly,
                    "metadata": reading.metadata
                })
            
            return monitoring_data
            
        except Exception as e:
            logger.error(f"Error getting monitoring data: {e}")
            return []
    
    async def _get_compliance_data(self, 
                                 query: str, 
                                 intent: Dict[str, Any],
                                 user: User, 
                                 db: Session,
                                 context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get relevant compliance data"""
        try:
            # Build query
            assessments_query = db.query(ComplianceAssessment).join(ComplianceRequirement)
            
            # Apply facility filter
            if context and context.get("facility_id"):
                assessments_query = assessments_query.filter(
                    ComplianceAssessment.facility_id == context["facility_id"]
                )
            
            # Get recent assessments
            assessments = assessments_query.order_by(
                ComplianceAssessment.assessment_date.desc()
            ).limit(20).all()
            
            # Format data
            compliance_data = []
            for assessment in assessments:
                compliance_data.append({
                    "requirement_id": assessment.requirement_id,
                    "facility_id": assessment.facility_id,
                    "assessment_date": assessment.assessment_date.isoformat(),
                    "status": assessment.status,
                    "compliance_score": assessment.compliance_score,
                    "risk_score": assessment.risk_score,
                    "findings": assessment.findings
                })
            
            return compliance_data
            
        except Exception as e:
            logger.error(f"Error getting compliance data: {e}")
            return []
    
    async def _generate_response(self, 
                               query: str, 
                               intent: Dict[str, Any],
                               relevant_data: Dict[str, Any],
                               user_id: int,
                               db: Session) -> Dict[str, Any]:
        """Generate intelligent response based on query and data"""
        try:
            if not self.llm:
                return self._generate_fallback_response(query, intent, relevant_data)
            
            # Create context from relevant data
            context = self._create_context(relevant_data)
            
            # Create prompt based on intent
            prompt = self._create_prompt(query, intent, context)
            
            # Generate response
            response = await self.llm.agenerate([prompt])
            
            # Parse and format response
            ai_response = response.generations[0][0].text.strip()
            
            # Add analysis and insights
            analysis = self._analyze_data_for_insights(relevant_data, intent)
            
            return {
                "response": ai_response,
                "analysis": analysis,
                "data_summary": self._create_data_summary(relevant_data),
                "recommendations": self._generate_recommendations(relevant_data, intent),
                "visualization_suggestions": self._suggest_visualizations(relevant_data, intent)
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._generate_fallback_response(query, intent, relevant_data)
    
    def _create_context(self, relevant_data: Dict[str, Any]) -> str:
        """Create context string from relevant data"""
        context_parts = []
        
        # Add document context
        if relevant_data.get("documents"):
            doc_summaries = []
            for doc in relevant_data["documents"][:5]:  # Top 5 most relevant
                doc_summaries.append(f"- {doc['metadata'].get('title', 'Unknown')}: {doc['content'][:200]}...")
            context_parts.append("Relevant Documents:\n" + "\n".join(doc_summaries))
        
        # Add monitoring context
        if relevant_data.get("monitoring_data"):
            monitoring_summary = self._summarize_monitoring_data(relevant_data["monitoring_data"])
            context_parts.append(f"Monitoring Data:\n{monitoring_summary}")
        
        # Add compliance context
        if relevant_data.get("compliance_data"):
            compliance_summary = self._summarize_compliance_data(relevant_data["compliance_data"])
            context_parts.append(f"Compliance Data:\n{compliance_summary}")
        
        return "\n\n".join(context_parts)
    
    def _create_prompt(self, query: str, intent: Dict[str, Any], context: str) -> str:
        """Create appropriate prompt based on query intent"""
        
        base_prompt = f"""
You are an AI assistant specialized in Tailings Storage Facility (TSF) management. 
You help engineers, operators, and managers with data analysis, compliance, and decision-making.

User Query: {query}

Context Information:
{context}

Please provide a comprehensive, professional response that:
1. Directly addresses the user's question
2. Uses the provided context data when relevant
3. Provides actionable insights and recommendations
4. Maintains technical accuracy for TSF management
5. Suggests next steps or follow-up actions when appropriate

Response:"""

        # Add intent-specific instructions
        if intent["type"] == "monitoring":
            base_prompt += "\n\nFocus on monitoring data analysis, trends, and alerts."
        elif intent["type"] == "compliance":
            base_prompt += "\n\nFocus on compliance status, requirements, and regulatory implications."
        elif intent["type"] == "prediction":
            base_prompt += "\n\nFocus on trends, forecasting, and predictive insights."
        elif intent["type"] == "alert":
            base_prompt += "\n\nFocus on current alerts, risk assessment, and immediate actions."
        
        return base_prompt
    
    def _summarize_monitoring_data(self, monitoring_data: List[Dict[str, Any]]) -> str:
        """Create summary of monitoring data"""
        if not monitoring_data:
            return "No monitoring data available."
        
        # Group by station
        stations = {}
        for reading in monitoring_data:
            station_id = reading["station_id"]
            if station_id not in stations:
                stations[station_id] = []
            stations[station_id].append(reading)
        
        summary_parts = []
        for station_id, readings in stations.items():
            latest = readings[0]  # Most recent
            summary_parts.append(
                f"Station {station_id}: {latest['value']} {latest['unit']} "
                f"(Alert: {latest['alert_level']})"
            )
        
        return "\n".join(summary_parts)
    
    def _summarize_compliance_data(self, compliance_data: List[Dict[str, Any]]) -> str:
        """Create summary of compliance data"""
        if not compliance_data:
            return "No compliance data available."
        
        # Count by status
        status_counts = {}
        for assessment in compliance_data:
            status = assessment["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        
        summary_parts = []
        for status, count in status_counts.items():
            summary_parts.append(f"{status}: {count} assessments")
        
        return "Compliance Status: " + ", ".join(summary_parts)
    
    def _analyze_data_for_insights(self, relevant_data: Dict[str, Any], intent: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data to provide insights"""
        insights = {
            "trends": [],
            "anomalies": [],
            "risks": [],
            "opportunities": []
        }
        
        # Analyze monitoring data for trends and anomalies
        if relevant_data.get("monitoring_data"):
            monitoring_insights = self._analyze_monitoring_insights(relevant_data["monitoring_data"])
            insights.update(monitoring_insights)
        
        # Analyze compliance data for risks
        if relevant_data.get("compliance_data"):
            compliance_insights = self._analyze_compliance_insights(relevant_data["compliance_data"])
            insights.update(compliance_insights)
        
        return insights
    
    def _analyze_monitoring_insights(self, monitoring_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze monitoring data for insights"""
        insights = {
            "trends": [],
            "anomalies": [],
            "risks": []
        }
        
        if not monitoring_data:
            return insights
        
        # Group by station and parameter
        station_data = {}
        for reading in monitoring_data:
            station_id = reading["station_id"]
            if station_id not in station_data:
                station_data[station_id] = []
            station_data[station_id].append(reading)
        
        # Analyze each station
        for station_id, readings in station_data.items():
            # Check for anomalies
            anomalies = [r for r in readings if r["is_anomaly"]]
            if anomalies:
                insights["anomalies"].append(f"Station {station_id}: {len(anomalies)} anomalous readings")
            
            # Check for critical alerts
            critical = [r for r in readings if r["alert_level"] == "critical"]
            if critical:
                insights["risks"].append(f"Station {station_id}: {len(critical)} critical alerts")
        
        return insights
    
    def _analyze_compliance_insights(self, compliance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze compliance data for insights"""
        insights = {
            "risks": [],
            "opportunities": []
        }
        
        if not compliance_data:
            return insights
        
        # Count non-compliant assessments
        non_compliant = [a for a in compliance_data if a["status"] == "non_compliant"]
        if non_compliant:
            insights["risks"].append(f"{len(non_compliant)} non-compliant assessments found")
        
        # Check for low compliance scores
        low_scores = [a for a in compliance_data if a.get("compliance_score", 100) < 70]
        if low_scores:
            insights["risks"].append(f"{len(low_scores)} assessments with low compliance scores")
        
        return insights
    
    def _create_data_summary(self, relevant_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of available data"""
        summary = {
            "documents_count": len(relevant_data.get("documents", [])),
            "monitoring_readings_count": len(relevant_data.get("monitoring_data", [])),
            "compliance_assessments_count": len(relevant_data.get("compliance_data", [])),
            "data_sources": relevant_data.get("sources", [])
        }
        
        return summary
    
    def _generate_recommendations(self, relevant_data: Dict[str, Any], intent: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Add general recommendations based on intent
        if intent["type"] == "monitoring":
            recommendations.append("Review monitoring data trends regularly")
            recommendations.append("Set up automated alerts for critical thresholds")
        
        if intent["type"] == "compliance":
            recommendations.append("Schedule regular compliance assessments")
            recommendations.append("Maintain up-to-date documentation")
        
        if intent["type"] == "prediction":
            recommendations.append("Implement predictive maintenance schedules")
            recommendations.append("Monitor key performance indicators")
        
        return recommendations
    
    def _suggest_visualizations(self, relevant_data: Dict[str, Any], intent: Dict[str, Any]) -> List[str]:
        """Suggest appropriate visualizations"""
        visualizations = []
        
        if relevant_data.get("monitoring_data"):
            visualizations.extend([
                "Time series chart of monitoring readings",
                "Alert level distribution pie chart",
                "Station performance comparison"
            ])
        
        if relevant_data.get("compliance_data"):
            visualizations.extend([
                "Compliance status dashboard",
                "Risk score heatmap",
                "Assessment timeline"
            ])
        
        return visualizations
    
    def _generate_fallback_response(self, query: str, intent: Dict[str, Any], relevant_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback response when AI is not available"""
        return {
            "response": f"I understand you're asking about: {query}. I'm currently analyzing the available data to provide you with the most relevant information.",
            "analysis": {"trends": [], "anomalies": [], "risks": [], "opportunities": []},
            "data_summary": self._create_data_summary(relevant_data),
            "recommendations": ["Contact your system administrator to enable AI features"],
            "visualization_suggestions": []
        }
    
    async def add_document_to_knowledge_base(self, document_id: int, db: Session) -> bool:
        """Add a document to the AI knowledge base for future queries"""
        try:
            # Get document and chunks
            document = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
            if not document:
                return False
            
            chunks = db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).all()
            
            if not chunks:
                return False
            
            # Create documents for vector store
            documents = []
            for chunk in chunks:
                doc = Document(
                    page_content=chunk.chunk_text,
                    metadata={
                        "document_id": document.id,
                        "chunk_index": chunk.chunk_index,
                        "title": document.title,
                        "document_type": document.document_type,
                        "facility_id": document.facility_id
                    }
                )
                documents.append(doc)
            
            # Add to vector store
            if self.vector_store and documents:
                self.vector_store.add_documents(documents)
                logger.info(f"Added document {document_id} to knowledge base")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error adding document to knowledge base: {e}")
            return False
    
    async def get_query_history(self, user_id: int, db: Session, limit: int = 50) -> List[Dict[str, Any]]:
        """Get user's query history"""
        # This would typically query a query_history table
        # For now, return empty list
        return []
    
    async def save_query(self, user_id: int, query: str, response: Dict[str, Any], db: Session) -> bool:
        """Save query and response for history"""
        # This would typically save to a query_history table
        # For now, just log
        logger.info(f"Query saved for user {user_id}: {query[:100]}...")
        return True 
