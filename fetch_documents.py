#!/usr/bin/env python3
"""
Document Fetcher Script for Czech Municipal Notice Board Chatbot

This script automates the collection of documents from Czech municipal
notice boards via the NKOD (National Open Data Catalog).

Usage:
    python fetch_documents.py [options]

Options:
    --cities CITY1,CITY2    Specify cities to fetch (default: all supported)
    --force-refresh         Force re-download even if cached
    --verbose              Show detailed output
    --output-stats         Display collection statistics
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import List, Dict

# Import our main components
from main import NKODCollector, DocumentProcessor, SimpleVectorStore

# Supported cities
DEFAULT_CITIES = [
    "Děčín", 
    "Ústí nad Labem", 
    "Praha", 
    "Brno", 
    "Ostrava", 
    "Teplice",
    "Liberec", 
    "Plzeň",
]

class DocumentFetcher:
    """
    Automated document fetcher for municipal notice boards.
    
    Handles the complete pipeline from NKOD search to processed documents.
    """
    
    def __init__(self, verbose: bool = False, force_refresh: bool = False):
        self.verbose = verbose
        self.force_refresh = force_refresh
        self.collector = NKODCollector()
        self.processor = DocumentProcessor()
        self.stats = {
            'cities_processed': 0,
            'datasets_found': 0,
            'documents_downloaded': 0,
            'documents_processed': 0,
            'errors': 0,
            'start_time': datetime.now()
        }
    
    def log(self, message: str, level: str = "INFO"):
        """Log messages with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = f"[{timestamp}] {level}:"
        
        if level == "ERROR":
            print(f"❌ {prefix} {message}")
        elif level == "SUCCESS":
            print(f"✅ {prefix} {message}")
        elif level == "WARNING":
            print(f"⚠️  {prefix} {message}")
        else:
            print(f"ℹ️  {prefix} {message}")
    
    def fetch_city_documents(self, city: str) -> List[Dict]:
        """
        Fetch documents for a specific city.
        
        Args:
            city: Name of the city
            
        Returns:
            List of processed document dictionaries
        """
        self.log(f"Processing {city}...")
        self.stats['cities_processed'] += 1
        
        try:
            # Check cache first
            if not self.force_refresh:
                cached = self.collector.load_cached(city)
                if cached:
                    self.log(f"Using cached metadata for {city} ({len(cached)} datasets)")
                    metadata = cached
                else:
                    self.log(f"No cache found for {city}, fetching from NKOD...")
                    metadata = self.collector.search_boards(city)
                    if metadata:
                        self.collector.cache_results(city, metadata)
            else:
                self.log(f"Force refresh: fetching {city} from NKOD...")
                metadata = self.collector.search_boards(city)
                if metadata:
                    self.collector.cache_results(city, metadata)
            
            if not metadata:
                self.log(f"No datasets found for {city}", "WARNING")
                return []
            
            self.stats['datasets_found'] += len(metadata)
            self.log(f"Found {len(metadata)} datasets for {city}")
            
            # Process documents
            if self.verbose:
                self.log(f"Processing documents for {city}...")
            
            documents = self.processor.process_documents(metadata)
            self.stats['documents_downloaded'] += len([d for d in documents if d.get('filepath')])
            self.stats['documents_processed'] += len(documents)
            
            self.log(f"Processed {len(documents)} documents for {city}", "SUCCESS")
            return documents
            
        except Exception as e:
            self.stats['errors'] += 1
            self.log(f"Error processing {city}: {e}", "ERROR")
            return []
    
    def fetch_all_documents(self, cities: List[str]) -> List[Dict]:
        """
        Fetch documents for all specified cities.
        
        Args:
            cities: List of city names
            
        Returns:
            Combined list of all processed documents
        """
        print("=" * 60)
        print("CZECH MUNICIPAL DOCUMENT FETCHER")
        print("=" * 60)
        
        all_documents = []
        
        for city in cities:
            city_docs = self.fetch_city_documents(city)
            all_documents.extend(city_docs)
            
            if self.verbose and city_docs:
                self.log(f"Sample document from {city}: {city_docs[0].get('title', 'No title')[:50]}...")
        
        return all_documents
    
    def save_to_vector_store(self, documents: List[Dict]):
        """Save documents to the vector store"""
        self.log("Saving documents to vector store...")
        
        try:
            vector_store = SimpleVectorStore()
            
            # Load existing documents if not force refresh
            if not self.force_refresh:
                vector_store.load()
                existing_count = len(vector_store.documents)
                self.log(f"Loaded {existing_count} existing documents")
            
            # Add new documents
            vector_store.add_documents(documents)
            total_count = len(vector_store.documents)
            
            self.log(f"Vector store now contains {total_count} documents", "SUCCESS")
            
        except Exception as e:
            self.stats['errors'] += 1
            self.log(f"Error saving to vector store: {e}", "ERROR")
    
    def print_statistics(self):
        """Print collection statistics"""
        duration = datetime.now() - self.stats['start_time']
        
        print("\n" + "=" * 60)
        print("COLLECTION STATISTICS")
        print("=" * 60)
        print(f"📊 Cities processed:      {self.stats['cities_processed']}")
        print(f"📋 Datasets found:        {self.stats['datasets_found']}")
        print(f"📄 Documents downloaded:  {self.stats['documents_downloaded']}")
        print(f"✅ Documents processed:   {self.stats['documents_processed']}")
        print(f"❌ Errors encountered:    {self.stats['errors']}")
        print(f"⏱️  Total duration:        {str(duration).split('.')[0]}")
        
        if self.stats['documents_processed'] > 0:
            avg_per_city = self.stats['documents_processed'] / max(1, self.stats['cities_processed'])
            print(f"📈 Average docs per city: {avg_per_city:.1f}")
        
        print("=" * 60)
    
    def export_document_list(self, documents: List[Dict], filename: str = "document_list.json"):
        """Export document metadata to JSON file"""
        try:
            export_data = {
                'export_date': datetime.now().isoformat(),
                'total_documents': len(documents),
                'cities': list(set(doc.get('municipality', 'Unknown') for doc in documents)),
                'documents': [
                    {
                        'doc_id': doc.get('doc_id', ''),
                        'title': doc.get('title', ''),
                        'municipality': doc.get('municipality', ''),
                        'original_url': doc.get('original_url', ''),
                        'processed_at': doc.get('processed_at', ''),
                        'has_content': bool(doc.get('text', '').strip())
                    }
                    for doc in documents
                ]
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            self.log(f"Document list exported to {filename}", "SUCCESS")
            
        except Exception as e:
            self.log(f"Error exporting document list: {e}", "ERROR")


def parse_cities(city_string: str) -> List[str]:
    """Parse comma-separated city list"""
    if not city_string:
        return DEFAULT_CITIES
    
    cities = [city.strip() for city in city_string.split(',')]
    return [city for city in cities if city]


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Fetch documents from Czech municipal notice boards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python fetch_documents.py
    python fetch_documents.py --cities "Praha,Brno"
    python fetch_documents.py --force-refresh --verbose
    python fetch_documents.py --output-stats
        """
    )
    
    parser.add_argument(
        '--cities',
        type=str,
        help=f'Comma-separated list of cities (default: {", ".join(DEFAULT_CITIES)})'
    )
    
    parser.add_argument(
        '--force-refresh',
        action='store_true',
        help='Force re-download even if cached data exists'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed output during processing'
    )
    
    parser.add_argument(
        '--output-stats',
        action='store_true',
        help='Display detailed collection statistics'
    )
    
    parser.add_argument(
        '--export-list',
        type=str,
        metavar='FILENAME',
        help='Export document metadata to JSON file'
    )
    
    args = parser.parse_args()
    
    # Parse cities
    cities = parse_cities(args.cities)
    
    if not cities:
        print("❌ No valid cities specified")
        return 1
    
    # Create fetcher
    fetcher = DocumentFetcher(
        verbose=args.verbose,
        force_refresh=args.force_refresh
    )
    
    try:
        # Fetch documents
        documents = fetcher.fetch_all_documents(cities)
        
        if not documents:
            fetcher.log("No documents were collected", "WARNING")
            return 1
        
        # Save to vector store
        fetcher.save_to_vector_store(documents)
        
        # Export document list if requested
        if args.export_list:
            fetcher.export_document_list(documents, args.export_list)
        
        # Print statistics
        if args.output_stats or args.verbose:
            fetcher.print_statistics()
        
        fetcher.log(f"Document collection completed successfully! Total: {len(documents)} documents", "SUCCESS")
        return 0
        
    except KeyboardInterrupt:
        fetcher.log("Collection interrupted by user", "WARNING")
        return 1
    except Exception as e:
        fetcher.log(f"Unexpected error: {e}", "ERROR")
        return 1


if __name__ == "__main__":
    sys.exit(main())