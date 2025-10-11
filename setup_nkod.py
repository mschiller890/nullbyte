#!/usr/bin/env python3
"""
Setup script for NKOD integration
Initializes the database and fetches initial data
"""

import asyncio
import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nkod_integration import NKODDataManager

async def setup_nkod():
    """Setup NKOD integration"""
    print("🚀 Initializing NKOD (National Catalog of Open Data) integration...")
    
    try:
        # Create NKOD manager
        manager = NKODDataManager()
        
        print("📊 Fetching initial data from data.gov.cz...")
        success = await manager.refresh_data(limit=50)  # Start with 50 datasets
        
        if success:
            stats = manager.get_stats()
            print(f"✅ Successfully initialized with {stats.get('indexed_datasets', 0)} datasets")
            print(f"📁 Data stored in: {manager.data_dir}")
            print("🎉 NKOD integration is ready!")
        else:
            print("❌ Failed to fetch initial data")
            print("💡 You can try manual setup later using the refresh button")
            
    except Exception as e:
        print(f"❌ Error during setup: {e}")
        print("💡 NKOD integration will work without initial data")

def main():
    """Main setup function"""
    print("=" * 60)
    print("🇨🇿 NKOD (Národní katalog otevřených dat) Setup")
    print("=" * 60)
    
    # Run async setup
    asyncio.run(setup_nkod())
    
    print("\n" + "=" * 60)
    print("✨ Setup complete! Start your chatbot with: python main.py")
    print("🔍 Try asking about 'otevřená data', 'statistiky', or 'datasety'")
    print("=" * 60)

if __name__ == "__main__":
    main()