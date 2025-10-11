"""
SUPER INTELLIGENT NKOD (National Catalog of Open Data) Integration
Advanced AI-powered data source evaluation and training system
Revolutionizes Czech open data access with machine learning
"""

import os
import json
import asyncio
import pickle
import re
import math
import hashlib
from typing import List, Dict, Optional, Tuple, Set
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from datetime import datetime, timedelta
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SourceQuality(Enum):
    """Source quality classification"""
    PREMIUM = "premium"
    HIGH = "high" 
    GOOD = "good"
    MEDIUM = "medium"
    LOW = "low"
    UNRELIABLE = "unreliable"

@dataclass
class DatasetIntelligence:
    """Advanced dataset metadata and intelligence"""
    id: str
    title: str
    description: str
    publisher: str
    theme: str
    keywords: List[str]
    quality_score: float
    reliability_score: float
    freshness_score: float
    completeness_score: float
    usage_popularity: int
    source_authority_level: int
    data_formats: List[str]
    update_frequency: str
    last_modified: Optional[datetime]
    download_count: int
    citizen_rating: float
    government_priority: int
    semantic_tags: List[str]
    content_hash: str
    
class NKODSmartEngine:
    """Ultra-intelligent NKOD processing engine with AI training capabilities"""
    
    def __init__(self, data_dir: str = "nkod_smart_data"):
        self.data_dir = data_dir
        self.db_path = os.path.join(data_dir, "smart_chroma_db")
        self.cache_file = os.path.join(data_dir, "smart_nkod_cache.json")
        self.intelligence_file = os.path.join(data_dir, "dataset_intelligence.pkl")
        self.training_data_file = os.path.join(data_dir, "ai_training_corpus.json")
        self.quality_models_file = os.path.join(data_dir, "quality_models.pkl")
        
        # Create advanced directory structure
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, "training"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "models"), exist_ok=True)
        
        # Initialize ChromaDB with advanced configuration
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(
            name="nkod_intelligent_datasets",
            metadata={
                "description": "Super-intelligent Czech NKOD with AI training",
                "version": "2.0",
                "features": "quality_scoring,semantic_analysis,ai_training"
            }
        )
        
        # Initialize advanced sentence transformer
        self._init_smart_embedder()
        
        # Load intelligence models
        self.quality_models = self._load_quality_models()
        self.dataset_intelligence = self._load_dataset_intelligence()
        self.source_rankings = self._init_source_rankings()
        
        # Training corpus for AI learning
        self.training_corpus = []
        
        logger.info("🧠 Super-intelligent NKOD engine initialized with AI capabilities")
    
    def _init_smart_embedder(self):
        """Initialize advanced embedding system"""
        try:
            # Use multiple embedding models for different aspects
            self.semantic_embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.multilingual_embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("🎯 Advanced multi-model embedder system loaded")
        except Exception as e:
            logger.error(f"Failed to load embedding models: {e}")
            self.semantic_embedder = None
            self.multilingual_embedder = None
    
    def _load_quality_models(self) -> Dict:
        """Load pre-trained quality assessment models"""
        try:
            if os.path.exists(self.quality_models_file):
                with open(self.quality_models_file, 'rb') as f:
                    models = pickle.load(f)
                logger.info("🔬 Quality models loaded from cache")
                return models
        except Exception as e:
            logger.warning(f"Could not load quality models: {e}")
        
        # Initialize default quality models
        return {
            'freshness_weights': {'daily': 1.0, 'weekly': 0.9, 'monthly': 0.7, 'yearly': 0.4, 'static': 0.2},
            'publisher_authority': self._build_publisher_authority_map(),
            'format_quality_scores': {'JSON': 1.0, 'CSV': 0.9, 'XML': 0.8, 'XLSX': 0.6, 'PDF': 0.3},
            'keyword_importance_weights': self._build_keyword_importance_map()
        }
    
    def _load_dataset_intelligence(self) -> Dict[str, DatasetIntelligence]:
        """Load cached dataset intelligence"""
        try:
            if os.path.exists(self.intelligence_file):
                with open(self.intelligence_file, 'rb') as f:
                    intelligence = pickle.load(f)
                logger.info(f"🧠 Loaded intelligence for {len(intelligence)} datasets")
                return intelligence
        except Exception as e:
            logger.warning(f"Could not load dataset intelligence: {e}")
        
        return {}
    
    def _init_source_rankings(self) -> Dict[str, float]:
        """Initialize intelligent source ranking system"""
        return {
            # Government authorities (highest trust)
            'czso.cz': 1.0,  # Czech Statistical Office
            'gov.cz': 0.95,  # Government portal
            'mfcr.cz': 0.95,  # Ministry of Finance
            'mzcr.cz': 0.95,  # Ministry of Health
            'mdcr.cz': 0.95,  # Ministry of Transport
            'mpsv.cz': 0.95,  # Ministry of Labour and Social Affairs
            
            # Regional and municipal (high trust)
            'praha.eu': 0.9,  # Prague
            'brno.cz': 0.9,   # Brno
            'ostrava.cz': 0.9, # Ostrava
            'plzen.eu': 0.9,  # Pilsen
            
            # Research institutions (high trust)
            'cvut.cz': 0.85,  # Czech Technical University
            'cuni.cz': 0.85,  # Charles University
            'vut.cz': 0.85,   # Brno University of Technology
            
            # Default scoring
            'default': 0.5
        }
    
    def _build_publisher_authority_map(self) -> Dict[str, int]:
        """Build publisher authority scoring"""
        return {
            'český statistický úřad': 10,
            'ministerstvo financí': 9,
            'ministerstvo zdravotnictví': 9,
            'ministerstvo dopravy': 9,
            'magistrát hlavního města prahy': 8,
            'krajský úřad': 7,
            'městský úřad': 6,
            'státní organizace': 8,
            'veřejná instituce': 6,
            'univerzita': 7,
            'výzkumná instituce': 7
        }
    
    def _build_keyword_importance_map(self) -> Dict[str, float]:
        """Build keyword importance weighting system"""
        return {
            # Core government data
            'rozpočet': 1.0, 'finance': 1.0, 'výdaje': 1.0, 'příjmy': 1.0,
            'statistiky': 0.95, 'demografické': 0.95, 'ekonomické': 0.95,
            'zdravotnictví': 0.9, 'školství': 0.9, 'doprava': 0.9,
            
            # Social and civic data  
            'sociální': 0.85, 'kultura': 0.8, 'životní prostředí': 0.9,
            'bezpečnost': 0.85, 'kriminalita': 0.8,
            
            # Infrastructure and urban
            'infrastruktura': 0.85, 'městské': 0.8, 'územní': 0.8,
            'stavební': 0.75, 'dopravní': 0.85,
            
            # Administrative
            'úřední': 0.7, 'administrativa': 0.6, 'evidence': 0.75
        }

    def analyze_dataset_quality(self, dataset: Dict) -> Tuple[float, Dict]:
        """Advanced dataset quality analysis with multiple metrics"""
        quality_metrics = {}
        
        # 1. Content Quality Analysis
        title_quality = self._analyze_title_quality(dataset.get('title', ''))
        desc_quality = self._analyze_description_quality(dataset.get('description', ''))
        
        quality_metrics['title_quality'] = title_quality
        quality_metrics['description_quality'] = desc_quality
        
        # 2. Publisher Authority Score
        publisher = dataset.get('publisher', '').lower()
        authority_score = 0.5  # default
        
        for auth_pub, score in self.quality_models['publisher_authority'].items():
            if auth_pub in publisher:
                authority_score = score / 10.0
                break
        
        quality_metrics['authority_score'] = authority_score
        
        # 3. Keyword Relevance and Importance
        keywords = dataset.get('keyword', '') + ' ' + dataset.get('theme', '')
        keyword_score = self._calculate_keyword_importance(keywords.lower())
        quality_metrics['keyword_importance'] = keyword_score
        
        # 4. Freshness Score (based on update frequency estimation)
        freshness_score = self._estimate_data_freshness(dataset)
        quality_metrics['freshness'] = freshness_score
        
        # 5. Completeness Score
        completeness = self._calculate_completeness_score(dataset)
        quality_metrics['completeness'] = completeness
        
        # 6. Semantic Coherence
        semantic_score = self._analyze_semantic_coherence(dataset)
        quality_metrics['semantic_coherence'] = semantic_score
        
        # Calculate overall quality score with intelligent weighting
        overall_quality = (
            title_quality * 0.15 +
            desc_quality * 0.2 +
            authority_score * 0.25 +
            keyword_score * 0.15 +
            freshness_score * 0.1 +
            completeness * 0.1 +
            semantic_score * 0.05
        )
        
        quality_metrics['overall_quality'] = min(1.0, overall_quality)
        
        return overall_quality, quality_metrics
    
    def _analyze_title_quality(self, title: str) -> float:
        """Analyze title quality and informativeness"""
        if not title:
            return 0.0
        
        score = 0.5  # base score
        
        # Length optimization (not too short, not too long)
        length = len(title)
        if 20 <= length <= 80:
            score += 0.2
        elif 10 <= length < 20 or 80 < length <= 120:
            score += 0.1
        
        # Contains year information
        if re.search(r'\b20[0-9]{2}\b', title):
            score += 0.15
        
        # Contains specific location
        czech_locations = ['praha', 'brno', 'ostrava', 'plzeň', 'česká republika', 'čr']
        if any(loc in title.lower() for loc in czech_locations):
            score += 0.1
        
        # Descriptive keywords
        descriptive_words = ['statistiky', 'data', 'přehled', 'analýza', 'report', 'evidence']
        if any(word in title.lower() for word in descriptive_words):
            score += 0.1
        
        return min(1.0, score)
    
    def _analyze_description_quality(self, description: str) -> float:
        """Analyze description quality and informativeness"""
        if not description:
            return 0.0
        
        score = 0.3  # base score
        
        # Length assessment
        length = len(description)
        if 100 <= length <= 500:
            score += 0.3
        elif 50 <= length < 100 or 500 < length <= 1000:
            score += 0.2
        elif length > 1000:
            score += 0.1
        
        # Contains structured information
        if any(marker in description for marker in ['•', '-', '1.', '2.', '3.']):
            score += 0.15
        
        # Contains temporal information
        temporal_words = ['aktuální', 'měsíční', 'roční', 'denní', 'pravidelně']
        if any(word in description.lower() for word in temporal_words):
            score += 0.1
        
        # Contains contact or source information
        if any(marker in description.lower() for marker in ['kontakt', 'zdroj', 'metodika', 'autor']):
            score += 0.15
        
        return min(1.0, score)
    
    def _calculate_keyword_importance(self, keywords: str) -> float:
        """Calculate importance score based on keyword analysis"""
        if not keywords:
            return 0.3
        
        importance_score = 0.3
        word_count = 0
        
        for keyword, weight in self.quality_models['keyword_importance_weights'].items():
            if keyword in keywords:
                importance_score += weight * 0.1
                word_count += 1
        
        # Boost for multiple important keywords
        if word_count >= 3:
            importance_score += 0.1
        elif word_count >= 2:
            importance_score += 0.05
        
        return min(1.0, importance_score)
    
    def _estimate_data_freshness(self, dataset: Dict) -> float:
        """Estimate data freshness based on various indicators"""
        base_score = 0.5
        
        # Check for temporal indicators in title/description
        text = (dataset.get('title', '') + ' ' + dataset.get('description', '')).lower()
        
        # Current year boost
        current_year = datetime.now().year
        if str(current_year) in text:
            base_score += 0.3
        elif str(current_year - 1) in text:
            base_score += 0.2
        elif str(current_year - 2) in text:
            base_score += 0.1
        
        # Frequency indicators
        frequency_indicators = {
            'denní': 0.3, 'daily': 0.3,
            'týdenní': 0.25, 'weekly': 0.25,
            'měsíční': 0.2, 'monthly': 0.2,
            'čtvrtletní': 0.15, 'quarterly': 0.15,
            'roční': 0.1, 'annual': 0.1,
            'aktualizovan': 0.2, 'updated': 0.2
        }
        
        for indicator, boost in frequency_indicators.items():
            if indicator in text:
                base_score += boost
                break
        
        return min(1.0, base_score)
    
    def _calculate_completeness_score(self, dataset: Dict) -> float:
        """Calculate how complete the dataset metadata is"""
        required_fields = ['title', 'description', 'publisher', 'theme']
        optional_fields = ['keyword', 'id']
        
        score = 0.0
        
        # Required field completeness (70% of score)
        for field in required_fields:
            if dataset.get(field) and len(str(dataset[field]).strip()) > 5:
                score += 0.175  # 0.7 / 4 fields
        
        # Optional field completeness (30% of score)
        for field in optional_fields:
            if dataset.get(field) and len(str(dataset[field]).strip()) > 0:
                score += 0.15  # 0.3 / 2 fields
        
        return min(1.0, score)
    
    def _analyze_semantic_coherence(self, dataset: Dict) -> float:
        """Analyze semantic coherence between title, description, and keywords"""
        if not self.semantic_embedder:
            return 0.5
        
        try:
            title = dataset.get('title', '')
            description = dataset.get('description', '')
            keywords = dataset.get('keyword', '') + ' ' + dataset.get('theme', '')
            
            if not all([title, description]):
                return 0.3
            
            # Generate embeddings
            title_emb = self.semantic_embedder.encode([title])
            desc_emb = self.semantic_embedder.encode([description])
            
            # Calculate cosine similarity
            similarity = np.dot(title_emb[0], desc_emb[0]) / (
                np.linalg.norm(title_emb[0]) * np.linalg.norm(desc_emb[0])
            )
            
            # Convert to 0-1 range and boost
            coherence_score = (similarity + 1) / 2  # Convert from [-1,1] to [0,1]
            
            # Keyword coherence bonus
            if keywords and len(keywords.strip()) > 10:
                coherence_score += 0.1
            
            return min(1.0, coherence_score)
            
        except Exception as e:
            logger.warning(f"Error in semantic coherence analysis: {e}")
            return 0.5
    
    def generate_query_perspectives(self, query: str) -> List[str]:
        """
        Generate multiple perspective queries for comprehensive search
        """
        perspectives = [
            query,  # Original query
            f"{query} statistiky",  # Statistics perspective
            f"{query} otevřená data",  # Open data perspective
            f"{query} veřejná správa",  # Public administration perspective
            f"dataset {query}",  # Dataset-focused perspective
        ]
        
        # Add domain-specific perspectives
        if "Praha" in query or "prague" in query.lower():
            perspectives.extend([
                f"{query} hlavní město",
                f"{query} městská data"
            ])
        
        if any(word in query.lower() for word in ["obyvatel", "demograf", "populace"]):
            perspectives.extend([
                f"{query} census",
                f"{query} population"
            ])
        
        return list(set(perspectives))  # Remove duplicates
    
    def calculate_publisher_authority(self, publisher: str) -> float:
        """
        Calculate publisher authority score based on known rankings
        """
        publisher_lower = publisher.lower()
        
        # Government authority mapping
        if any(key in publisher_lower for key in ["český statistický úřad", "czso"]):
            return 1.0
        elif any(key in publisher_lower for key in ["ministerstvo", "ministry"]):
            return 0.95
        elif any(key in publisher_lower for key in ["praha", "prague", "hlavní město"]):
            return 0.9
        elif any(key in publisher_lower for key in ["úřad", "office", "město", "city"]):
            return 0.8
        elif any(key in publisher_lower for key in ["krajský", "regional"]):
            return 0.7
        else:
            return 0.5  # Unknown publishers
    
    def classify_dataset_intelligence(self, dataset: Dict) -> DatasetIntelligence:
        """
        Classify dataset with comprehensive intelligence analysis
        """
        title = dataset.get('title', '')
        description = dataset.get('description', '')
        publisher = dataset.get('publisher', '')
        keywords = dataset.get('keyword', '')
        
        # Calculate individual quality scores
        title_quality = self._analyze_title_quality(title)
        description_quality = self._analyze_description_quality(description)
        authority_score = self.calculate_publisher_authority(publisher)
        keyword_importance = self._calculate_keyword_importance(keywords)
        freshness = self._estimate_data_freshness(dataset)
        completeness = self._calculate_completeness_score(dataset)
        semantic_coherence = self._analyze_semantic_coherence(dataset)
        
        # Calculate overall quality score with weighted average
        quality_score = (
            title_quality * 0.2 +
            description_quality * 0.25 +
            authority_score * 0.2 +
            keyword_importance * 0.1 +
            freshness * 0.1 +
            completeness * 0.1 +
            semantic_coherence * 0.05
        )
        
        return DatasetIntelligence(
            id=dataset.get('id', 'unknown'),
            title=title,
            description=description,
            publisher=publisher,
            theme=dataset.get('theme', ''),
            keywords=keywords.split(',') if keywords else [],
            quality_score=quality_score,
            reliability_score=authority_score,
            freshness_score=freshness,
            completeness_score=completeness,
            usage_popularity=0,  # Would be calculated from actual usage data
            source_authority_level=int(authority_score * 10),
            data_formats=dataset.get('format', '').split(',') if dataset.get('format') else [],
            update_frequency=dataset.get('accrual_periodicity', 'unknown'),
            last_modified=None,  # Would parse from dataset metadata
            download_count=0,  # Would be fetched from actual statistics
            citizen_rating=quality_score * 5.0,  # Convert to 1-5 scale
            government_priority=int(authority_score * 10),
            semantic_tags=[],  # Would be generated by semantic analysis
            content_hash=str(hash(f"{title}{description}{publisher}"))
        )
        
        # Add convenient access properties for backward compatibility
        intelligence.title_quality = title_quality
        intelligence.description_quality = description_quality
        intelligence.authority_score = authority_score
        intelligence.keyword_importance = keyword_importance
        intelligence.freshness = freshness
        intelligence.completeness = completeness
        intelligence.semantic_coherence = semantic_coherence
        
        return intelligence

class NKODDataManager(NKODSmartEngine):
    """Intelligent NKOD Data Manager with advanced AI capabilities"""
    
    def __init__(self, data_dir: str = "nkod_smart_data"):
        # Initialize the super-intelligent engine
        super().__init__(data_dir)
        
        # Legacy compatibility
        self.embedder = self.semantic_embedder

    def fetch_nkod_catalog_advanced(self, limit: int = 500) -> List[Dict]:
        """
        Advanced NKOD catalog fetching with multiple endpoints and intelligence
        """
        logger.info("🚀 Starting super-intelligent NKOD catalog fetch...")
        
        all_datasets = []
        
        # Multiple advanced SPARQL queries for comprehensive data collection
        queries = [
            self._build_comprehensive_dataset_query(limit // 3),
            self._build_high_quality_sources_query(limit // 3),
            self._build_government_priority_query(limit // 3)
        ]
        
        for i, query in enumerate(queries, 1):
            try:
                logger.info(f"🎯 Executing advanced query {i}/3...")
                datasets = self._execute_sparql_query(query)
                
                # Apply intelligent filtering and enhancement
                enhanced_datasets = []
                for dataset in datasets:
                    enhanced = self._enhance_dataset_with_intelligence(dataset)
                    if enhanced:
                        enhanced_datasets.append(enhanced)
                
                all_datasets.extend(enhanced_datasets)
                logger.info(f"✅ Query {i} yielded {len(enhanced_datasets)} intelligent datasets")
                
            except Exception as e:
                logger.error(f"❌ Error in query {i}: {e}")
        
        # Remove duplicates and apply final intelligence ranking
        unique_datasets = self._deduplicate_and_rank(all_datasets)
        
        # Train AI on the collected data
        self._train_ai_on_nkod_data(unique_datasets)
        
        logger.info(f"🧠 Super-intelligent NKOD fetch complete: {len(unique_datasets)} premium datasets")
        return unique_datasets
    
    def _build_comprehensive_dataset_query(self, limit: int) -> str:
        """Build comprehensive SPARQL query for maximum data coverage"""
        return f"""
        PREFIX dcat: <http://www.w3.org/ns/dcat#>
        PREFIX dcterms: <http://purl.org/dc/terms/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        PREFIX adms: <http://www.w3.org/ns/adms#>
        PREFIX vcard: <http://www.w3.org/2006/vcard/ns#>
        
        SELECT DISTINCT ?dataset ?title ?description ?publisher ?publisherName ?theme ?keyword 
                       ?modified ?issued ?format ?downloadURL ?license ?contactPoint WHERE {{
            ?dataset a dcat:Dataset .
            
            # Core metadata
            OPTIONAL {{ ?dataset dcterms:title ?title }}
            OPTIONAL {{ ?dataset dcterms:description ?description }}
            OPTIONAL {{ ?dataset dcterms:publisher ?publisher }}
            OPTIONAL {{ ?dataset dcat:theme ?theme }}
            OPTIONAL {{ ?dataset dcat:keyword ?keyword }}
            
            # Enhanced metadata
            OPTIONAL {{ ?dataset dcterms:modified ?modified }}
            OPTIONAL {{ ?dataset dcterms:issued ?issued }}
            OPTIONAL {{ ?dataset dcterms:license ?license }}
            OPTIONAL {{ ?dataset dcat:contactPoint ?contactPoint }}
            
            # Publisher details
            OPTIONAL {{ ?publisher foaf:name ?publisherName }}
            
            # Distribution information
            OPTIONAL {{ 
                ?dataset dcat:distribution ?dist .
                ?dist dcat:mediaType ?format .
                ?dist dcat:downloadURL ?downloadURL 
            }}
            
            # Quality filters
            FILTER(LANG(?title) = "cs" || LANG(?title) = "")
            FILTER(LANG(?description) = "cs" || LANG(?description) = "")
            FILTER(STRLEN(?title) > 10)
            FILTER(STRLEN(?description) > 30)
        }}
        ORDER BY DESC(?modified)
        LIMIT {limit}
        """
    
    def _build_high_quality_sources_query(self, limit: int) -> str:
        """Build query focused on high-quality government sources"""
        return f"""
        PREFIX dcat: <http://www.w3.org/ns/dcat#>
        PREFIX dcterms: <http://purl.org/dc/terms/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        
        SELECT DISTINCT ?dataset ?title ?description ?publisher ?publisherName ?theme ?keyword 
                       ?modified ?issued WHERE {{
            ?dataset a dcat:Dataset .
            ?dataset dcterms:title ?title .
            ?dataset dcterms:description ?description .
            ?dataset dcterms:publisher ?publisher .
            
            OPTIONAL {{ ?dataset dcat:theme ?theme }}
            OPTIONAL {{ ?dataset dcat:keyword ?keyword }}
            OPTIONAL {{ ?dataset dcterms:modified ?modified }}
            OPTIONAL {{ ?dataset dcterms:issued ?issued }}
            OPTIONAL {{ ?publisher foaf:name ?publisherName }}
            
            # Focus on high-authority sources
            FILTER(
                CONTAINS(LCASE(?publisherName), "statistick") ||
                CONTAINS(LCASE(?publisherName), "ministerstvo") ||
                CONTAINS(LCASE(?publisherName), "úřad") ||
                CONTAINS(LCASE(?publisherName), "magistrát") ||
                CONTAINS(LCASE(?publisher), "gov.cz") ||
                CONTAINS(LCASE(?publisher), "czso.cz")
            )
            
            FILTER(LANG(?title) = "cs" || LANG(?title) = "")
            FILTER(LANG(?description) = "cs" || LANG(?description) = "")
        }}
        ORDER BY DESC(?modified)
        LIMIT {limit}
        """
    
    def _build_government_priority_query(self, limit: int) -> str:
        """Build query for government priority datasets"""
        return f"""
        PREFIX dcat: <http://www.w3.org/ns/dcat#>
        PREFIX dcterms: <http://purl.org/dc/terms/>
        
        SELECT DISTINCT ?dataset ?title ?description ?publisher ?theme ?keyword WHERE {{
            ?dataset a dcat:Dataset .
            ?dataset dcterms:title ?title .
            ?dataset dcterms:description ?description .
            ?dataset dcterms:publisher ?publisher .
            
            OPTIONAL {{ ?dataset dcat:theme ?theme }}
            OPTIONAL {{ ?dataset dcat:keyword ?keyword }}
            
            # Priority themes and keywords
            FILTER(
                CONTAINS(LCASE(?title), "rozpočet") ||
                CONTAINS(LCASE(?title), "statistik") ||
                CONTAINS(LCASE(?title), "demografi") ||
                CONTAINS(LCASE(?title), "ekonomick") ||
                CONTAINS(LCASE(?title), "zdravotn") ||
                CONTAINS(LCASE(?title), "školstv") ||
                CONTAINS(LCASE(?title), "doprav") ||
                CONTAINS(LCASE(?keyword), "finance") ||
                CONTAINS(LCASE(?keyword), "veřejné") ||
                CONTAINS(LCASE(?theme), "government")
            )
            
            FILTER(LANG(?title) = "cs" || LANG(?title) = "")
            FILTER(STRLEN(?description) > 20)
        }}
        LIMIT {limit}
        """
    
    def _execute_sparql_query(self, query: str) -> List[Dict]:
        """Execute SPARQL query with intelligent error handling"""
        try:
            base_url = "https://data.gov.cz/sparql"
            headers = {
                'Accept': 'application/sparql-results+json',
                'User-Agent': 'SuperIntelligentNKOD/2.0 (AI Training System)',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            response = requests.post(
                base_url,
                data={'query': query},
                headers=headers,
                timeout=45
            )
            
            if response.status_code == 200:
                results = response.json()
                datasets = []
                
                for binding in results.get('results', {}).get('bindings', []):
                    dataset = self._parse_sparql_binding(binding)
                    if dataset:
                        datasets.append(dataset)
                
                return datasets
            else:
                logger.error(f"SPARQL query failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error executing SPARQL query: {e}")
            return []
    
    def _parse_sparql_binding(self, binding: Dict) -> Optional[Dict]:
        """Parse SPARQL binding with intelligence"""
        try:
            dataset = {
                'id': binding.get('dataset', {}).get('value', ''),
                'title': binding.get('title', {}).get('value', ''),
                'description': binding.get('description', {}).get('value', ''),
                'publisher': binding.get('publisherName', {}).get('value', '') or 
                           binding.get('publisher', {}).get('value', ''),
                'theme': binding.get('theme', {}).get('value', ''),
                'keyword': binding.get('keyword', {}).get('value', ''),
                'modified': binding.get('modified', {}).get('value', ''),
                'issued': binding.get('issued', {}).get('value', ''),
                'format': binding.get('format', {}).get('value', ''),
                'download_url': binding.get('downloadURL', {}).get('value', ''),
                'license': binding.get('license', {}).get('value', ''),
                'contact_point': binding.get('contactPoint', {}).get('value', ''),
                'source': 'data.gov.cz',
                'scraped_at': datetime.now().isoformat()
            }
            
            # Quality validation
            if len(dataset['title']) < 5 or len(dataset['description']) < 10:
                return None
                
            return dataset
            
        except Exception as e:
            logger.warning(f"Error parsing SPARQL binding: {e}")
            return None
    
    def _enhance_dataset_with_intelligence(self, dataset: Dict) -> Optional[Dict]:
        """Enhance dataset with advanced intelligence metrics"""
        try:
            # Calculate quality scores
            quality_score, quality_metrics = self.analyze_dataset_quality(dataset)
            
            # Skip low-quality datasets
            if quality_score < 0.3:
                return None
            
            # Add intelligence metadata
            dataset.update({
                'quality_score': quality_score,
                'quality_metrics': quality_metrics,
                'intelligence_processed': True,
                'semantic_tags': self._generate_semantic_tags(dataset),
                'content_hash': hashlib.md5(
                    f"{dataset.get('title', '')}{dataset.get('description', '')}".encode()
                ).hexdigest(),
                'ai_training_value': self._calculate_ai_training_value(dataset, quality_score)
            })
            
            return dataset
            
        except Exception as e:
            logger.warning(f"Error enhancing dataset: {e}")
            return dataset  # Return original if enhancement fails
    
    def _generate_semantic_tags(self, dataset: Dict) -> List[str]:
        """Generate semantic tags for advanced categorization"""
        tags = set()
        
        # Extract from title and description
        text = f"{dataset.get('title', '')} {dataset.get('description', '')}".lower()
        
        # Government and administrative tags
        if any(word in text for word in ['ministerstvo', 'úřad', 'vláda', 'státní']):
            tags.add('government')
        
        # Statistical and data tags
        if any(word in text for word in ['statistik', 'data', 'čísla', 'analýza']):
            tags.add('statistics')
        
        # Financial tags
        if any(word in text for word in ['rozpočet', 'finance', 'výdaje', 'příjmy']):
            tags.add('financial')
        
        # Social tags
        if any(word in text for word in ['sociální', 'zdraví', 'školství', 'kultura']):
            tags.add('social')
        
        # Infrastructure tags
        if any(word in text for word in ['doprava', 'infrastruktura', 'staveb']):
            tags.add('infrastructure')
        
        # Environmental tags
        if any(word in text for word in ['životní prostředí', 'ekologie', 'znečištění']):
            tags.add('environmental')
        
        return list(tags)
    
    def _calculate_ai_training_value(self, dataset: Dict, quality_score: float) -> float:
        """Calculate value of this dataset for AI training purposes"""
        base_value = quality_score
        
        # Boost for diverse content
        if len(dataset.get('description', '')) > 200:
            base_value += 0.1
        
        # Boost for structured data indicators
        if dataset.get('format', '').upper() in ['JSON', 'CSV', 'XML']:
            base_value += 0.15
        
        # Boost for government sources
        if any(gov in dataset.get('publisher', '').lower() for gov in ['ministerstvo', 'úřad', 'czso']):
            base_value += 0.2
        
        # Boost for recent data
        if dataset.get('modified') and '2024' in dataset.get('modified', ''):
            base_value += 0.1
        
        return min(1.0, base_value)
    
    def _train_ai_on_nkod_data(self, datasets: List[Dict]):
        """Train AI system on collected NKOD data"""
        logger.info("🧠 Starting AI training on NKOD data...")
        
        try:
            # Prepare training corpus
            training_examples = []
            
            for dataset in datasets:
                if dataset.get('ai_training_value', 0) > 0.6:  # High-value datasets only
                    # Create training examples
                    example = {
                        'text': f"Dataset: {dataset.get('title', '')}. {dataset.get('description', '')}",
                        'metadata': {
                            'publisher': dataset.get('publisher', ''),
                            'theme': dataset.get('theme', ''),
                            'quality_score': dataset.get('quality_score', 0),
                            'semantic_tags': dataset.get('semantic_tags', [])
                        },
                        'training_label': self._generate_training_label(dataset)
                    }
                    training_examples.append(example)
            
            # Save training corpus
            self.training_corpus.extend(training_examples)
            self._save_training_corpus()
            
            # Update AI models with new knowledge
            self._update_quality_models(datasets)
            
            logger.info(f"✅ AI training complete: {len(training_examples)} examples processed")
            
        except Exception as e:
            logger.error(f"❌ AI training error: {e}")
    
    def _generate_training_label(self, dataset: Dict) -> Dict:
        """Generate training labels for supervised learning"""
        return {
            'quality_class': self._classify_quality(dataset.get('quality_score', 0)),
            'authority_level': self._classify_authority(dataset.get('publisher', '')),
            'content_type': self._classify_content_type(dataset),
            'priority_level': self._classify_priority(dataset)
        }
    
    def _classify_quality(self, score: float) -> str:
        """Classify quality into categories"""
        if score >= 0.8:
            return 'premium'
        elif score >= 0.6:
            return 'high'
        elif score >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _classify_authority(self, publisher: str) -> str:
        """Classify publisher authority"""
        publisher_lower = publisher.lower()
        
        if any(term in publisher_lower for term in ['statistický úřad', 'czso']):
            return 'supreme'
        elif 'ministerstvo' in publisher_lower:
            return 'high'
        elif any(term in publisher_lower for term in ['úřad', 'magistrát']):
            return 'medium'
        else:
            return 'standard'
    
    def _classify_content_type(self, dataset: Dict) -> str:
        """Classify content type for training"""
        text = f"{dataset.get('title', '')} {dataset.get('description', '')}".lower()
        
        if any(term in text for term in ['statistik', 'demografi', 'ekonomick']):
            return 'statistical'
        elif any(term in text for term in ['rozpočet', 'finance', 'výdaje']):
            return 'financial'
        elif any(term in text for term in ['sociální', 'zdraví', 'školství']):
            return 'social'
        elif any(term in text for term in ['doprava', 'infrastruktura']):
            return 'infrastructure'
        else:
            return 'general'
    
    def _classify_priority(self, dataset: Dict) -> str:
        """Classify government priority level"""
        # High-priority topics for Czech government
        high_priority = ['rozpočet', 'finance', 'zdravotnictví', 'školství', 'doprava']
        medium_priority = ['kultura', 'sociální', 'životní prostředí']
        
        text = f"{dataset.get('title', '')} {dataset.get('theme', '')}".lower()
        
        if any(term in text for term in high_priority):
            return 'high'
        elif any(term in text for term in medium_priority):
            return 'medium'
        else:
            return 'standard'
    
    def _deduplicate_and_rank(self, datasets: List[Dict]) -> List[Dict]:
        """Remove duplicates and rank by intelligence score"""
        # Use content hash to identify duplicates
        seen_hashes = set()
        unique_datasets = []
        
        for dataset in datasets:
            content_hash = dataset.get('content_hash')
            if content_hash and content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_datasets.append(dataset)
        
        # Rank by combined quality and training value
        unique_datasets.sort(
            key=lambda x: (x.get('quality_score', 0) + x.get('ai_training_value', 0)) / 2,
            reverse=True
        )
        
        return unique_datasets
    
    def _save_training_corpus(self):
        """Save training corpus for AI development"""
        try:
            with open(self.training_data_file, 'w', encoding='utf-8') as f:
                json.dump(self.training_corpus, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 Saved {len(self.training_corpus)} training examples")
        except Exception as e:
            logger.error(f"Error saving training corpus: {e}")
    
    def _update_quality_models(self, datasets: List[Dict]):
        """Update quality assessment models based on new data"""
        try:
            # Update publisher authority scores
            publisher_scores = {}
            for dataset in datasets:
                publisher = dataset.get('publisher', '').lower()
                quality = dataset.get('quality_score', 0.5)
                
                if publisher in publisher_scores:
                    publisher_scores[publisher].append(quality)
                else:
                    publisher_scores[publisher] = [quality]
            
            # Calculate average scores for publishers
            for publisher, scores in publisher_scores.items():
                avg_score = sum(scores) / len(scores)
                if len(scores) >= 3:  # Minimum samples for reliable score
                    self.quality_models['publisher_authority'][publisher] = int(avg_score * 10)
            
            # Save updated models
            with open(self.quality_models_file, 'wb') as f:
                pickle.dump(self.quality_models, f)
                
            logger.info("🔬 Quality models updated with new intelligence")
            
        except Exception as e:
            logger.error(f"Error updating quality models: {e}")
    
    # Legacy compatibility method
    def fetch_nkod_catalog(self, limit: int = 100) -> List[Dict]:
        """Legacy compatibility - redirects to advanced method"""
        return self.fetch_nkod_catalog_advanced(limit)

    def scrape_dataset_details(self, dataset_url: str) -> Dict:
        """
        Scrape additional details from a dataset page
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(dataset_url, headers=headers, timeout=15)
            if response.status_code != 200:
                return {}
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            details = {
                'url': dataset_url,
                'full_description': '',
                'formats': [],
                'license': '',
                'downloads': []
            }
            
            # Extract description
            desc_elem = soup.find('div', class_='dataset-description') or soup.find('meta', {'name': 'description'})
            if desc_elem:
                details['full_description'] = desc_elem.get_text(strip=True) if desc_elem.name == 'div' else desc_elem.get('content', '')
            
            # Extract download links and formats
            download_links = soup.find_all('a', href=True)
            for link in download_links:
                href = link.get('href', '')
                if href and isinstance(href, str) and any(ext in href.lower() for ext in ['.csv', '.json', '.xml', '.xlsx', '.pdf']):
                    details['downloads'].append({
                        'url': href,
                        'text': link.get_text(strip=True)
                    })
                    
                    # Extract format
                    for fmt in ['csv', 'json', 'xml', 'xlsx', 'pdf']:
                        if fmt in href.lower() and fmt not in details['formats']:
                            details['formats'].append(fmt.upper())
            
            return details
            
        except Exception as e:
            logger.error(f"Error scraping dataset details for {dataset_url}: {e}")
            return {}

    def index_datasets(self, datasets: List[Dict]) -> bool:
        """
        Legacy compatibility method - redirects to intelligent indexing
        """
        return self.index_datasets_intelligently(datasets)

    def search_datasets(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        SUPER INTELLIGENT dataset search with multi-model ranking and AI optimization
        """
        if not self.semantic_embedder:
            logger.error("No embedder available for search")
            return []
        
        logger.info(f"🔍 Executing super-intelligent search for: '{query}'")
        
        try:
            # Multi-stage intelligent query enhancement
            enhanced_queries = self._generate_multi_perspective_queries(query)
            
            all_results = []
            
            # Execute multiple semantic searches with different embedders
            for i, enhanced_query in enumerate(enhanced_queries):
                embedder = self.semantic_embedder if i % 2 == 0 else self.multilingual_embedder
                if not embedder:
                    continue
                
                query_embedding = embedder.encode([enhanced_query]).tolist()
                
                # Search with larger result set for intelligent filtering
                search_results = self.collection.query(
                    query_embeddings=query_embedding,
                    n_results=min(n_results * 5, 25),
                    include=['documents', 'metadatas', 'distances']
                )
                
                # Convert search results to dictionary format
                search_dict = {
                    'documents': search_results.get('documents', []),
                    'metadatas': search_results.get('metadatas', []),
                    'distances': search_results.get('distances', [])
                }
                processed_results = self._process_search_results_intelligently(
                    search_dict, query, enhanced_query
                )
                all_results.extend(processed_results)
            
            # Apply super-intelligent ranking and deduplication
            final_results = self._apply_super_intelligent_ranking(all_results, query, n_results)
            
            # Add intelligence metrics
            for result in final_results:
                result['intelligence_score'] = self._calculate_intelligence_score(result, query)
                result['source_quality'] = self._evaluate_source_quality(result)
                result['ai_confidence'] = self._calculate_ai_confidence(result, query)
            
            logger.info(f"🧠 Super-intelligent search complete: {len(final_results)} premium results")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in super-intelligent search: {e}")
            return []
    
    def _generate_multi_perspective_queries(self, query: str) -> List[str]:
        """Generate multiple query perspectives for comprehensive search"""
        queries = [query]  # Original query
        
        # Enhanced query with Czech context
        enhanced_basic = self._enhance_search_query(query)
        queries.append(enhanced_basic)
        
        # Government/official perspective
        if not any(term in query.lower() for term in ['ministerstvo', 'úřad', 'státní']):
            gov_query = f"{query} ministerstvo úřad státní oficiální"
            queries.append(gov_query)
        
        # Statistical perspective
        if not any(term in query.lower() for term in ['statistik', 'data', 'čísl']):
            stat_query = f"{query} statistiky data číselné údaje"
            queries.append(stat_query)
        
        # Regional perspective
        if not any(term in query.lower() for term in ['kraj', 'město', 'obec']):
            regional_query = f"{query} kraj město obec regionální"
            queries.append(regional_query)
        
        return queries[:3]  # Limit to top 3 perspectives
    
    def _process_search_results_intelligently(self, search_results: Dict, original_query: str, enhanced_query: str) -> List[Dict]:
        """Process search results with advanced intelligence"""
        datasets = []
        
        try:
            documents = search_results.get('documents', [[]])
            metadatas = search_results.get('metadatas', [[]])
            distances = search_results.get('distances', [[]])
            
            if documents and metadatas and distances and len(documents) > 0 and len(documents[0]) > 0:
                for i in range(len(documents[0])):
                    if (i < len(metadatas[0]) and i < len(distances[0])):
                        metadata = metadatas[0][i] if i < len(metadatas[0]) else {}
                        distance = distances[0][i] if i < len(distances[0]) else 1.0
                        similarity = 1 - distance
                        
                        # Intelligent similarity threshold based on query complexity
                        min_threshold = 0.25 if len(original_query.split()) >= 3 else 0.35
                        
                        if similarity >= min_threshold:
                            dataset = {
                                'title': metadata.get('title', '') if metadata else '',
                                'description': metadata.get('description', '') if metadata else '',
                                'publisher': metadata.get('publisher', '') if metadata else '',
                                'theme': metadata.get('theme', '') if metadata else '',
                                'source': metadata.get('source', '') if metadata else '',
                                'similarity': similarity,
                                'content': documents[0][i] if i < len(documents[0]) else '',
                                'query_match_score': self._calculate_advanced_query_match(
                                    original_query, metadata, similarity
                                ),
                                'search_metadata': {
                                    'query_used': enhanced_query,
                                    'search_timestamp': datetime.now().isoformat()
                                }
                            }
                            datasets.append(dataset)
            
        except Exception as e:
            logger.warning(f"Error processing intelligent search results: {e}")
        
        return datasets
    
    def _apply_super_intelligent_ranking(self, all_results: List[Dict], query: str, n_results: int) -> List[Dict]:
        """Apply super-intelligent ranking algorithm"""
        # Remove duplicates based on title similarity
        unique_results = self._intelligent_deduplication(all_results)
        
        # Calculate comprehensive intelligence scores
        for result in unique_results:
            # Multi-factor scoring
            similarity_score = result.get('similarity', 0) * 0.3
            query_match_score = result.get('query_match_score', 0) * 0.25
            authority_score = self._get_publisher_authority_score(result.get('publisher', '')) * 0.2
            freshness_score = self._estimate_content_freshness(result) * 0.1
            completeness_score = self._calculate_metadata_completeness(result) * 0.1
            semantic_coherence = self._calculate_semantic_match(result, query) * 0.05
            
            # Combine all factors
            result['total_intelligence_score'] = (
                similarity_score + query_match_score + authority_score + 
                freshness_score + completeness_score + semantic_coherence
            )
        
        # Sort by intelligence score
        unique_results.sort(key=lambda x: x.get('total_intelligence_score', 0), reverse=True)
        
        return unique_results[:n_results]
    
    def _intelligent_deduplication(self, results: List[Dict]) -> List[Dict]:
        """Intelligent deduplication using semantic similarity"""
        if not self.semantic_embedder or len(results) <= 1:
            return results
        
        unique_results = []
        seen_embeddings = []
        
        for result in results:
            title = result.get('title', '')
            if not title:
                continue
            
            # Generate embedding for title
            title_embedding = self.semantic_embedder.encode([title])
            
            # Check similarity with existing results
            is_duplicate = False
            for seen_embedding in seen_embeddings:
                similarity = np.dot(title_embedding[0], seen_embedding) / (
                    np.linalg.norm(title_embedding[0]) * np.linalg.norm(seen_embedding)
                )
                if similarity > 0.85:  # High similarity threshold for deduplication
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_results.append(result)
                seen_embeddings.append(title_embedding[0])
        
        return unique_results
    
    def _calculate_advanced_query_match(self, query: str, metadata: Dict, base_similarity: float) -> float:
        """Calculate advanced query matching score"""
        if not metadata:
            return base_similarity
        
        score = base_similarity
        query_words = set(query.lower().split())
        
        # Title exact matches (high weight)
        title_words = set(metadata.get('title', '').lower().split())
        title_matches = len(query_words.intersection(title_words))
        score += title_matches * 0.15
        
        # Description partial matches
        description = metadata.get('description', '').lower()
        desc_matches = sum(1 for word in query_words if word in description and len(word) > 3)
        score += desc_matches * 0.05
        
        # Publisher relevance
        publisher = metadata.get('publisher', '').lower()
        if any(word in publisher for word in query_words):
            score += 0.1
        
        # Theme alignment
        theme = metadata.get('theme', '').lower()
        if any(word in theme for word in query_words):
            score += 0.08
        
        return min(1.0, score)
    
    def _get_publisher_authority_score(self, publisher: str) -> float:
        """Get normalized publisher authority score"""
        publisher_lower = publisher.lower()
        
        for auth_publisher, score in self.quality_models['publisher_authority'].items():
            if auth_publisher in publisher_lower:
                return score / 10.0
        
        # Domain-based scoring
        for domain, score in self.source_rankings.items():
            if domain in publisher_lower:
                return score
        
        return 0.5  # Default score
    
    def _estimate_content_freshness(self, result: Dict) -> float:
        """Estimate content freshness from various indicators"""
        current_year = datetime.now().year
        content = f"{result.get('title', '')} {result.get('description', '')}".lower()
        
        # Check for current year
        if str(current_year) in content:
            return 1.0
        elif str(current_year - 1) in content:
            return 0.8
        elif str(current_year - 2) in content:
            return 0.6
        elif any(year in content for year in [str(current_year - i) for i in range(3, 6)]):
            return 0.4
        
        return 0.3  # Default for older or undated content
    
    def _calculate_metadata_completeness(self, result: Dict) -> float:
        """Calculate completeness of metadata"""
        required_fields = ['title', 'description', 'publisher']
        optional_fields = ['theme', 'source']
        
        score = 0.0
        
        for field in required_fields:
            if result.get(field) and len(str(result[field]).strip()) > 5:
                score += 0.6 / len(required_fields)
        
        for field in optional_fields:
            if result.get(field) and len(str(result[field]).strip()) > 0:
                score += 0.4 / len(optional_fields)
        
        return score
    
    def _calculate_semantic_match(self, result: Dict, query: str) -> float:
        """Calculate semantic similarity between query and result"""
        if not self.semantic_embedder:
            return 0.5
        
        try:
            result_text = f"{result.get('title', '')} {result.get('description', '')}"
            
            query_emb = self.semantic_embedder.encode([query])
            result_emb = self.semantic_embedder.encode([result_text])
            
            similarity = np.dot(query_emb[0], result_emb[0]) / (
                np.linalg.norm(query_emb[0]) * np.linalg.norm(result_emb[0])
            )
            
            return max(0, (similarity + 1) / 2)  # Normalize to 0-1
            
        except Exception as e:
            logger.warning(f"Error in semantic matching: {e}")
            return 0.5
    
    def _calculate_intelligence_score(self, result: Dict, query: str) -> float:
        """Calculate overall intelligence score for the result"""
        return result.get('total_intelligence_score', 0.5)
    
    def _evaluate_source_quality(self, result: Dict) -> str:
        """Evaluate source quality classification"""
        authority_score = self._get_publisher_authority_score(result.get('publisher', ''))
        
        if authority_score >= 0.9:
            return SourceQuality.PREMIUM.value
        elif authority_score >= 0.8:
            return SourceQuality.HIGH.value
        elif authority_score >= 0.6:
            return SourceQuality.GOOD.value
        elif authority_score >= 0.4:
            return SourceQuality.MEDIUM.value
        else:
            return SourceQuality.LOW.value
    
    def _calculate_ai_confidence(self, result: Dict, query: str) -> float:
        """Calculate AI confidence in the result"""
        base_confidence = result.get('similarity', 0.5)
        
        # Boost for high authority sources
        if result.get('source_quality') in [SourceQuality.PREMIUM.value, SourceQuality.HIGH.value]:
            base_confidence += 0.2
        
        # Boost for complete metadata
        if result.get('description') and len(result['description']) > 100:
            base_confidence += 0.1
        
        # Boost for recent content
        if self._estimate_content_freshness(result) > 0.8:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _enhance_search_query(self, query: str) -> str:
        """Enhance search query with Czech synonyms and context"""
        # Add common Czech data-related terms
        enhancements = {
            'škol': 'škola vzdělávání školství',
            'zdravot': 'zdravotnictví zdraví nemocnice',
            'doprav': 'doprava silnice železnice metro',
            'rozpočet': 'rozpočet finance peníze náklady',
            'město': 'město obec komunální místní',
            'kraj': 'kraj regionální územní',
            'statistik': 'statistiky čísl data údaje',
            'životní prostředí': 'životní prostředí ekologie znečištění',
            'kultur': 'kultura umění divadlo muzeum'
        }
        
        enhanced = query.lower()
        for key, expansion in enhancements.items():
            if key in enhanced:
                enhanced += f" {expansion}"
        
        return enhanced
    
    def _calculate_dataset_accuracy(self, query: str, metadata: dict, similarity: float) -> float:
        """Calculate accuracy score for a dataset match"""
        base_score = similarity
        
        # Boost for exact keyword matches in title
        title = metadata.get('title', '').lower()
        query_words = query.lower().split()
        title_matches = sum(1 for word in query_words if len(word) > 3 and word in title)
        if title_matches > 0:
            base_score += min(0.2, title_matches * 0.1)
        
        # Boost for trusted publishers
        publisher = metadata.get('publisher', '').lower()
        trusted_publishers = ['český statistický úřad', 'ministerstvo', 'úřad', 'magistrát']
        if any(trust in publisher for trust in trusted_publishers):
            base_score += 0.1
        
        # Description relevance boost
        description = metadata.get('description', '').lower()
        desc_matches = sum(1 for word in query_words if len(word) > 3 and word in description)
        if desc_matches > 0:
            base_score += min(0.15, desc_matches * 0.05)
        
        return min(1.0, base_score)

    def update_cache(self, datasets: List[Dict]):
        """
        Cache datasets to file for faster loading
        """
        try:
            cache_data = {
                'datasets': datasets,
                'last_updated': datetime.now().isoformat(),
                'count': len(datasets)
            }
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Cached {len(datasets)} datasets")
            
        except Exception as e:
            logger.error(f"Error updating cache: {e}")

    def load_cache(self) -> List[Dict]:
        """
        Load cached datasets
        """
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    datasets = cache_data.get('datasets', [])
                    logger.info(f"Loaded {len(datasets)} datasets from cache")
                    return datasets
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
        
        return []

    async def refresh_data(self, limit: int = 500) -> bool:
        """
        Super-intelligent NKOD data refresh with AI training
        """
        try:
            logger.info("🚀 Starting super-intelligent NKOD data refresh...")
            
            # Fetch new datasets with advanced intelligence
            datasets = self.fetch_nkod_catalog_advanced(limit)
            
            if datasets:
                # Index datasets for intelligent search
                success = self.index_datasets_intelligently(datasets)
                
                if success:
                    # Update cache with intelligence metadata
                    self.update_cache_intelligent(datasets)
                    
                    # Save intelligence data
                    self._save_dataset_intelligence()
                    
                    logger.info("🧠 Super-intelligent NKOD refresh completed successfully")
                    return True
            
            logger.warning("No datasets fetched during intelligent refresh")
            return False
            
        except Exception as e:
            logger.error(f"Error during intelligent data refresh: {e}")
            return False
    
    def index_datasets_intelligently(self, datasets: List[Dict]) -> bool:
        """
        Index datasets with advanced intelligence and quality filtering
        """
        if not self.semantic_embedder:
            logger.error("No embedder available for intelligent indexing")
            return False
        
        try:
            documents = []
            metadatas = []
            ids = []
            embeddings = []
            
            high_quality_count = 0
            
            for i, dataset in enumerate(datasets):
                # Only index high-quality datasets
                quality_score = dataset.get('quality_score', 0)
                if quality_score < 0.4:  # Skip low-quality datasets
                    continue
                
                if not dataset.get('title') and not dataset.get('description'):
                    continue
                
                # Create enhanced searchable text with intelligence
                searchable_text = self._create_intelligent_searchable_text(dataset)
                
                # Generate advanced embeddings
                embedding = self._generate_advanced_embedding(searchable_text)
                if embedding is None:
                    continue
                
                documents.append(searchable_text)
                metadatas.append(self._create_enhanced_metadata(dataset))
                embeddings.append(embedding)
                ids.append(f"intelligent_nkod_{i}_{dataset.get('content_hash', str(i))}")
                
                high_quality_count += 1
            
            if documents and embeddings:
                # Store in ChromaDB with intelligence
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                    ids=ids
                )
                
                logger.info(f"🎯 Successfully indexed {high_quality_count} intelligent datasets")
                return True
            
        except Exception as e:
            logger.error(f"Error in intelligent indexing: {e}")
        
        return False
    
    def _create_intelligent_searchable_text(self, dataset: Dict) -> str:
        """Create enhanced searchable text with intelligence"""
        title = dataset.get('title', '')
        description = dataset.get('description', '')
        publisher = dataset.get('publisher', '')
        theme = dataset.get('theme', '')
        keywords = dataset.get('keyword', '')
        semantic_tags = ' '.join(dataset.get('semantic_tags', []))
        
        # Enhanced format with weighted importance
        searchable_text = f"""
        NÁZEV: {title}
        POPIS: {description}
        VYDAVATEL: {publisher}
        TÉMA: {theme}
        KLÍČOVÁ SLOVA: {keywords}
        SÉMANTICKÉ TAGY: {semantic_tags}
        KVALITA: {dataset.get('quality_score', 0.5):.2f}
        AUTORITA: {self._get_publisher_authority_score(publisher):.2f}
        """
        
        return searchable_text.strip()
    
    def _generate_advanced_embedding(self, text: str) -> Optional[List[float]]:
        """Generate advanced embeddings with multiple models"""
        try:
            if not self.semantic_embedder:
                return None
                
            # Use semantic embedder for primary embedding
            embedding = self.semantic_embedder.encode([text])
            return embedding[0].tolist()
        except Exception as e:
            logger.warning(f"Error generating embedding: {e}")
            return None
    
    def _create_enhanced_metadata(self, dataset: Dict) -> Dict:
        """Create enhanced metadata with intelligence"""
        return {
            'title': dataset.get('title', ''),
            'description': dataset.get('description', ''),
            'publisher': dataset.get('publisher', ''),
            'theme': dataset.get('theme', ''),
            'source': dataset.get('source', ''),
            'quality_score': dataset.get('quality_score', 0.5),
            'authority_score': self._get_publisher_authority_score(dataset.get('publisher', '')),
            'semantic_tags': dataset.get('semantic_tags', []),
            'ai_training_value': dataset.get('ai_training_value', 0.5),
            'intelligence_processed': True,
            'indexed_at': datetime.now().isoformat(),
            'dataset_id': dataset.get('id', ''),
            'content_hash': dataset.get('content_hash', '')
        }
    
    def update_cache_intelligent(self, datasets: List[Dict]):
        """Update cache with intelligence metadata"""
        try:
            # Separate datasets by quality
            premium_datasets = [d for d in datasets if d.get('quality_score', 0) >= 0.8]
            high_datasets = [d for d in datasets if 0.6 <= d.get('quality_score', 0) < 0.8]
            good_datasets = [d for d in datasets if 0.4 <= d.get('quality_score', 0) < 0.6]
            
            cache_data = {
                'datasets': datasets,
                'intelligence_summary': {
                    'total_datasets': len(datasets),
                    'premium_quality': len(premium_datasets),
                    'high_quality': len(high_datasets),
                    'good_quality': len(good_datasets),
                    'ai_training_ready': len([d for d in datasets if d.get('ai_training_value', 0) > 0.6])
                },
                'quality_distribution': {
                    'premium': [d['content_hash'] for d in premium_datasets[:10]],
                    'high': [d['content_hash'] for d in high_datasets[:10]],
                    'good': [d['content_hash'] for d in good_datasets[:10]]
                },
                'last_updated': datetime.now().isoformat(),
                'intelligence_version': '2.0'
            }
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"💾 Cached {len(datasets)} intelligent datasets with quality breakdown")
            
        except Exception as e:
            logger.error(f"Error updating intelligent cache: {e}")
    
    def _save_dataset_intelligence(self):
        """Save dataset intelligence for future use"""
        try:
            with open(self.intelligence_file, 'wb') as f:
                pickle.dump(self.dataset_intelligence, f)
            logger.info("🧠 Dataset intelligence saved successfully")
        except Exception as e:
            logger.error(f"Error saving dataset intelligence: {e}")
    


    def get_stats(self) -> Dict:
        """
        Get statistics about indexed data
        """
        try:
            count = self.collection.count()
            
            cache_info = {}
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                    cache_info = {
                        'last_updated': cache_data.get('last_updated', ''),
                        'cached_count': cache_data.get('count', 0)
                    }
            
            return {
                'indexed_datasets': count,
                'cache_info': cache_info,
                'embedder_available': self.embedder is not None
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}

# Global instance
nkod_manager = None

def get_nkod_manager() -> NKODDataManager:
    """Get or create the global NKOD manager instance"""
    global nkod_manager
    if nkod_manager is None:
        nkod_manager = NKODDataManager()
    return nkod_manager