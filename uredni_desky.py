"""
Úřední Desky - Official Municipal Boards Data Collection and Processing
Integrates with NKOD to find, download, and process municipal board documents
"""

import os
import json
import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import List, Dict, Optional
import logging
from datetime import datetime
import hashlib
import urllib.parse
import time
import fitz  # PyMuPDF for PDF processing
import chromadb
from sentence_transformers import SentenceTransformer
import asyncio

logger = logging.getLogger(__name__)

class UredniDeskyManager:
    """Manages collection and processing of úřední desky (official boards) data"""
    
    def __init__(self, data_dir: str = "uredni_desky_data"):
        self.data_dir = data_dir
        self.documents_dir = os.path.join(data_dir, "documents")
        self.processed_dir = os.path.join(data_dir, "processed")
        self.db_path = os.path.join(data_dir, "uredni_desky_db")
        
        # Create directories
        for dir_path in [self.data_dir, self.documents_dir, self.processed_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Initialize ChromaDB for RAG
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(
            name="uredni_desky_documents",
            metadata={"description": "Czech Municipal Board Documents for RAG"}
        )
        
        # Initialize embedder
        try:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer loaded for úřední desky processing")
        except Exception as e:
            logger.error(f"Failed to load embedder: {e}")
            self.embedder = None
    
    def search_uredni_desky_in_nkod(self, mesto_obec: Optional[str] = None) -> List[Dict]:
        """Search for úřední desky datasets in NKOD"""
        try:
            base_url = "https://data.gov.cz/sparql"
            
            # SPARQL query for úřední desky
            if mesto_obec:
                filter_clause = f'FILTER(CONTAINS(LCASE(?title), "úřední deska") && CONTAINS(LCASE(?title), "{mesto_obec.lower()}"))'
            else:
                filter_clause = 'FILTER(CONTAINS(LCASE(?title), "úřední deska"))'
            
            query = f"""
            PREFIX dcat: <http://www.w3.org/ns/dcat#>
            PREFIX dcterms: <http://purl.org/dc/terms/>
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            
            SELECT ?dataset ?title ?description ?publisher ?downloadURL ?accessURL WHERE {{
                ?dataset a dcat:Dataset .
                ?dataset dcterms:title ?title .
                OPTIONAL {{ ?dataset dcterms:description ?description }}
                OPTIONAL {{ ?dataset dcterms:publisher ?publisher }}
                OPTIONAL {{ 
                    ?dataset dcat:distribution ?dist .
                    ?dist dcat:downloadURL ?downloadURL 
                }}
                OPTIONAL {{ 
                    ?dataset dcat:distribution ?dist .
                    ?dist dcat:accessURL ?accessURL 
                }}
                {filter_clause}
                FILTER(LANG(?title) = "cs" || LANG(?title) = "")
            }}
            LIMIT 20
            """
            
            response = requests.post(
                base_url,
                data={'query': query},
                headers={
                    'Accept': 'application/sparql-results+json',
                    'User-Agent': 'UredniDeskyBot/1.0'
                },
                timeout=30
            )
            
            if response.status_code == 200:
                results = response.json()
                datasets = []
                
                for binding in results.get('results', {}).get('bindings', []):
                    dataset = {
                        'id': binding.get('dataset', {}).get('value', ''),
                        'title': binding.get('title', {}).get('value', ''),
                        'description': binding.get('description', {}).get('value', ''),
                        'publisher': binding.get('publisher', {}).get('value', ''),
                        'download_url': binding.get('downloadURL', {}).get('value', ''),
                        'access_url': binding.get('accessURL', {}).get('value', ''),
                        'found_at': datetime.now().isoformat()
                    }
                    datasets.append(dataset)
                
                logger.info(f"Found {len(datasets)} úřední desky datasets for {mesto_obec or 'all municipalities'}")
                return datasets
            else:
                logger.error(f"NKOD query failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching úřední desky in NKOD: {e}")
            return []
    
    def scrape_municipal_board_documents(self, municipality_url: str, municipality_name: str) -> List[Dict]:
        """Scrape documents from a municipal board website"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(municipality_url, headers=headers, timeout=15)
            if response.status_code != 200:
                logger.error(f"Failed to access {municipality_url}")
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            documents = []
            
            # Common patterns for úřední deska documents
            document_patterns = [
                'a[href*=".pdf"]',
                'a[href*="dokument"]', 
                'a[href*="priloha"]',
                '.document-link',
                '.attachment'
            ]
            
            for pattern in document_patterns:
                links = soup.select(pattern)
                for link in links:
                    href_attr = link.get('href', '')
                    href = str(href_attr) if href_attr else ''
                    title = link.get_text(strip=True) or str(link.get('title', ''))
                    
                    if href and title:
                        # Convert relative URLs to absolute
                        if href.startswith('/'):
                            href = urllib.parse.urljoin(municipality_url, href)
                        elif not href.startswith('http'):
                            href = urllib.parse.urljoin(municipality_url, href)
                        
                        documents.append({
                            'title': title,
                            'url': href,
                            'municipality': municipality_name,
                            'scraped_at': datetime.now().isoformat(),
                            'file_hash': hashlib.md5(href.encode()).hexdigest()
                        })
            
            # Remove duplicates
            seen_urls = set()
            unique_documents = []
            for doc in documents:
                if doc['url'] not in seen_urls:
                    seen_urls.add(doc['url'])
                    unique_documents.append(doc)
            
            logger.info(f"Found {len(unique_documents)} documents for {municipality_name}")
            return unique_documents
            
        except Exception as e:
            logger.error(f"Error scraping {municipality_url}: {e}")
            return []
    
    def download_document(self, doc_info: Dict) -> Optional[str]:
        """Download a document and return local file path"""
        try:
            response = requests.get(doc_info['url'], timeout=30)
            if response.status_code != 200:
                return None
            
            # Determine file extension
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' in content_type:
                extension = '.pdf'
            elif 'doc' in content_type:
                extension = '.doc'
            elif 'txt' in content_type:
                extension = '.txt'
            else:
                # Guess from URL
                url_lower = doc_info['url'].lower()
                if '.pdf' in url_lower:
                    extension = '.pdf'
                elif '.doc' in url_lower:
                    extension = '.doc'
                else:
                    extension = '.pdf'  # Default to PDF
            
            # Save file
            filename = f"{doc_info['file_hash']}{extension}"
            filepath = os.path.join(self.documents_dir, filename)
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded: {doc_info['title']} -> {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error downloading {doc_info['url']}: {e}")
            return None
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_text = page.get_text()
                if isinstance(page_text, str):
                    text += page_text + "\n"
            
            doc.close()
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def process_document(self, doc_info: Dict, filepath: str) -> Dict:
        """Process a downloaded document and extract text"""
        try:
            # Extract text based on file type
            if filepath.endswith('.pdf'):
                text_content = self.extract_text_from_pdf(filepath)
            elif filepath.endswith('.txt'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    text_content = f.read()
            else:
                # For other formats, try simple text extraction
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        text_content = f.read()
                except:
                    text_content = ""
            
            # Create processed document
            processed_doc = {
                'original_info': doc_info,
                'file_path': filepath,
                'text_content': text_content,
                'word_count': len(text_content.split()),
                'char_count': len(text_content),
                'processed_at': datetime.now().isoformat(),
                'has_content': len(text_content.strip()) > 100
            }
            
            # Save processed document
            processed_filename = f"{doc_info['file_hash']}_processed.json"
            processed_filepath = os.path.join(self.processed_dir, processed_filename)
            
            with open(processed_filepath, 'w', encoding='utf-8') as f:
                json.dump(processed_doc, f, ensure_ascii=False, indent=2)
            
            return processed_doc
            
        except Exception as e:
            logger.error(f"Error processing document {filepath}: {e}")
            return {}
    
    def add_to_rag_database(self, processed_doc: Dict):
        """Add processed document to RAG database"""
        if not self.embedder or not processed_doc.get('has_content'):
            return
        
        try:
            text_content = processed_doc['text_content']
            doc_info = processed_doc['original_info']
            
            # Split text into chunks for better RAG performance
            chunks = self._split_text_into_chunks(text_content)
            
            documents = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_info['file_hash']}_chunk_{i}"
                
                documents.append(chunk)
                metadatas.append({
                    'title': doc_info['title'],
                    'municipality': doc_info['municipality'],
                    'url': doc_info['url'],
                    'chunk_index': i,
                    'processed_at': processed_doc['processed_at'],
                    'file_hash': doc_info['file_hash']
                })
                ids.append(chunk_id)
            
            if documents:
                # Generate embeddings and store
                embeddings = self.embedder.encode(documents).tolist()
                
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                    ids=ids
                )
                
                logger.info(f"Added {len(documents)} chunks to RAG database for {doc_info['title']}")
        
        except Exception as e:
            logger.error(f"Error adding to RAG database: {e}")
    
    def _split_text_into_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks for RAG"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk = ' '.join(chunk_words)
            if len(chunk.strip()) > 100:  # Only add meaningful chunks
                chunks.append(chunk.strip())
        
        return chunks
    
    def search_rag_documents(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search RAG database for relevant document chunks"""
        if not self.embedder:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedder.encode([query]).tolist()
            
            # Search ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            documents = []
            docs = results.get('documents')
            if docs and len(docs) > 0 and docs[0]:
                for i in range(len(docs[0])):
                    # Safe access to ChromaDB results
                    metadatas = results.get('metadatas', [[]])
                    distances = results.get('distances', [[]])
                    
                    doc = {
                        'content': docs[0][i],
                        'metadata': metadatas[0][i] if metadatas and len(metadatas[0]) > i else {},
                        'similarity': 1 - distances[0][i] if distances and len(distances[0]) > i else 0
                    }
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error searching RAG documents: {e}")
            return []
    
    async def collect_municipality_data(self, municipality_name: str) -> Dict:
        """Complete pipeline: search NKOD, scrape, download, process, and add to RAG"""
        logger.info(f"Starting data collection for {municipality_name}")
        
        results = {
            'municipality': municipality_name,
            'nkod_datasets': [],
            'scraped_documents': [],
            'downloaded_files': [],
            'processed_documents': [],
            'rag_entries': 0,
            'started_at': datetime.now().isoformat()
        }
        
        try:
            # 1. Search NKOD for municipality's úřední deska datasets
            nkod_datasets = self.search_uredni_desky_in_nkod(municipality_name)
            results['nkod_datasets'] = nkod_datasets
            
            # 2. Process each dataset
            for dataset in nkod_datasets:
                # Try to get direct access to documents
                urls_to_try = [
                    dataset.get('access_url'),
                    dataset.get('download_url'),
                    dataset.get('id')
                ]
                
                for url in urls_to_try:
                    if url and url.startswith('http'):
                        # Scrape documents from this URL
                        documents = self.scrape_municipal_board_documents(url, municipality_name)
                        results['scraped_documents'].extend(documents)
                        
                        # Download and process documents (limit to prevent overload)
                        for doc in documents[:5]:  # Limit to 5 docs per source
                            # Download
                            filepath = self.download_document(doc)
                            if filepath:
                                results['downloaded_files'].append(filepath)
                                
                                # Process
                                processed = self.process_document(doc, filepath)
                                if processed:
                                    results['processed_documents'].append(processed)
                                    
                                    # Add to RAG
                                    self.add_to_rag_database(processed)
                                    results['rag_entries'] += 1
                            
                            # Add small delay to be respectful
                            await asyncio.sleep(1)
                        
                        break  # Found working URL for this dataset
            
            results['completed_at'] = datetime.now().isoformat()
            logger.info(f"Completed data collection for {municipality_name}: {results['rag_entries']} documents in RAG")
            
        except Exception as e:
            logger.error(f"Error in municipality data collection: {e}")
            results['error'] = str(e)
        
        return results
    
    def get_rag_stats(self) -> Dict:
        """Get statistics about RAG database"""
        try:
            count = self.collection.count()
            return {
                'total_chunks': count,
                'embedder_available': self.embedder is not None,
                'database_path': self.db_path
            }
        except Exception as e:
            return {'error': str(e)}

# Global instance
uredni_desky_manager = None

def get_uredni_desky_manager() -> UredniDeskyManager:
    """Get or create the global úřední desky manager"""
    global uredni_desky_manager
    if uredni_desky_manager is None:
        uredni_desky_manager = UredniDeskyManager()
    return uredni_desky_manager