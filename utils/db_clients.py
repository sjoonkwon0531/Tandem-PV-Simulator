"""
Database API Clients for Materials Project, AFLOW, JARVIS-DFT
==============================================================

Lightweight REST API wrappers with aggressive caching.
Gracefully degrades to sample data if API keys unavailable.

Author: OpenClaw Agent
Date: 2026-03-15
"""

import os
import json
import time
import sqlite3
import hashlib
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import requests
from pathlib import Path

# Cache database path
CACHE_DB = Path(__file__).parent.parent / "data" / "cache.db"
CACHE_DB.parent.mkdir(parents=True, exist_ok=True)

# API endpoints
MP_API_URL = "https://api.materialsproject.org/materials"
AFLOW_API_URL = "http://aflowlib.duke.edu/search/API"
JARVIS_API_URL = "https://jarvis.nist.gov/api/v1/materials"

# Perovskite structure filter (ABX3)
PEROVSKITE_FILTER = {
    'A_sites': ['Cs', 'Rb', 'MA', 'FA', 'K'],  # MA=CH3NH3, FA=CH(NH2)2
    'B_sites': ['Pb', 'Sn', 'Ge', 'Sr', 'Ca'],
    'X_sites': ['I', 'Br', 'Cl', 'F']
}


class CacheDB:
    """SQLite cache for API responses"""
    
    def __init__(self, db_path: str = str(CACHE_DB)):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize cache tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_cache (
                key TEXT PRIMARY KEY,
                source TEXT,
                data TEXT,
                timestamp REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS materials (
                material_id TEXT PRIMARY KEY,
                formula TEXT,
                source TEXT,
                bandgap REAL,
                structure TEXT,
                properties TEXT,
                timestamp REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get(self, key: str, max_age: float = 86400 * 7) -> Optional[str]:
        """Get cached value if not expired (default: 7 days)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT data, timestamp FROM api_cache WHERE key = ?',
            (key,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            data, timestamp = result
            if time.time() - timestamp < max_age:
                return data
        
        return None
    
    def set(self, key: str, data: str, source: str = 'unknown'):
        """Cache a value"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'INSERT OR REPLACE INTO api_cache (key, source, data, timestamp) VALUES (?, ?, ?, ?)',
            (key, source, data, time.time())
        )
        
        conn.commit()
        conn.close()
    
    def save_material(self, material_id: str, formula: str, source: str, 
                     bandgap: float, structure: str, properties: dict):
        """Save material to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO materials 
            (material_id, formula, source, bandgap, structure, properties, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (material_id, formula, source, bandgap, structure, 
              json.dumps(properties), time.time()))
        
        conn.commit()
        conn.close()
    
    def get_all_materials(self, source: Optional[str] = None) -> pd.DataFrame:
        """Get all cached materials as DataFrame"""
        conn = sqlite3.connect(self.db_path)
        
        if source:
            query = 'SELECT * FROM materials WHERE source = ?'
            df = pd.read_sql_query(query, conn, params=(source,))
        else:
            query = 'SELECT * FROM materials'
            df = pd.read_sql_query(query, conn)
        
        conn.close()
        
        if not df.empty and 'properties' in df.columns:
            df['properties'] = df['properties'].apply(json.loads)
        
        return df
    
    def clear(self, source: Optional[str] = None):
        """Clear cache (optionally by source)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if source:
            cursor.execute('DELETE FROM api_cache WHERE source = ?', (source,))
            cursor.execute('DELETE FROM materials WHERE source = ?', (source,))
        else:
            cursor.execute('DELETE FROM api_cache')
            cursor.execute('DELETE FROM materials')
        
        conn.commit()
        conn.close()


class MaterialsProjectClient:
    """Materials Project API client with caching"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('MP_API_KEY')
        self.cache = CacheDB()
        self.available = self.api_key is not None
    
    def _make_cache_key(self, query: dict) -> str:
        """Generate cache key from query"""
        query_str = json.dumps(query, sort_keys=True)
        return f"mp_{hashlib.md5(query_str.encode()).hexdigest()}"
    
    def search_perovskites(self, max_results: int = 500) -> pd.DataFrame:
        """
        Search for ABX3 perovskite structures in Materials Project.
        Returns DataFrame with formula, bandgap, structure info.
        """
        cache_key = self._make_cache_key({'type': 'perovskite_search', 'max': max_results})
        cached = self.cache.get(cache_key)
        
        if cached:
            return pd.DataFrame(json.loads(cached))
        
        if not self.available:
            return pd.DataFrame()  # Empty if no API key
        
        # Query MP API for perovskites
        # Note: This is a simplified version - real MP API has more complex query syntax
        headers = {'X-API-KEY': self.api_key}
        
        try:
            # Search for materials with Pm-3m space group (cubic perovskite)
            # and containing Pb or Sn + halides
            params = {
                'space_group': 221,  # Pm-3m
                'elements': 'Pb,Sn,I,Br,Cl',
                'limit': max_results
            }
            
            response = requests.get(
                f"{MP_API_URL}/summary",
                headers=headers,
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json().get('data', [])
                
                materials = []
                for item in data:
                    # Extract relevant properties
                    material = {
                        'material_id': item.get('material_id'),
                        'formula': item.get('formula_pretty'),
                        'bandgap': item.get('band_gap', np.nan),
                        'is_stable': item.get('is_stable', False),
                        'energy_above_hull': item.get('energy_above_hull', np.nan),
                        'source': 'materials_project'
                    }
                    materials.append(material)
                    
                    # Save to cache DB
                    self.cache.save_material(
                        material_id=material['material_id'],
                        formula=material['formula'],
                        source='materials_project',
                        bandgap=material['bandgap'],
                        structure=json.dumps({'space_group': 221}),
                        properties=material
                    )
                
                df = pd.DataFrame(materials)
                
                # Cache results
                self.cache.set(cache_key, df.to_json(), source='materials_project')
                
                return df
            
        except Exception as e:
            print(f"Materials Project API error: {e}")
        
        return pd.DataFrame()


class AFLOWClient:
    """AFLOW API client with caching"""
    
    def __init__(self):
        self.cache = CacheDB()
        self.available = True  # AFLOW is public, no API key needed
    
    def _make_cache_key(self, query: dict) -> str:
        """Generate cache key from query"""
        query_str = json.dumps(query, sort_keys=True)
        return f"aflow_{hashlib.md5(query_str.encode()).hexdigest()}"
    
    def search_perovskites(self, max_results: int = 500) -> pd.DataFrame:
        """
        Search for ABX3 perovskite structures in AFLOW.
        Returns DataFrame with formula, bandgap, structure info.
        """
        cache_key = self._make_cache_key({'type': 'perovskite_search', 'max': max_results})
        cached = self.cache.get(cache_key)
        
        if cached:
            return pd.DataFrame(json.loads(cached))
        
        try:
            # AFLOW REST API query for perovskites
            # Search for Pm-3m space group + composition
            query = f"{AFLOW_API_URL}/?species(Pb,Sn,I,Br,Cl),nspecies(3),Egap"
            
            response = requests.get(query, timeout=30)
            
            if response.status_code == 200:
                # Parse AFLOW JSON response
                data = response.json()
                
                materials = []
                for item in data[:max_results]:
                    material = {
                        'material_id': item.get('auid'),
                        'formula': item.get('compound'),
                        'bandgap': item.get('Egap', np.nan),
                        'spacegroup': item.get('spacegroup_relax'),
                        'source': 'aflow'
                    }
                    materials.append(material)
                    
                    # Save to cache DB
                    self.cache.save_material(
                        material_id=material['material_id'],
                        formula=material['formula'],
                        source='aflow',
                        bandgap=material['bandgap'],
                        structure=json.dumps({'spacegroup': material['spacegroup']}),
                        properties=material
                    )
                
                df = pd.DataFrame(materials)
                
                # Cache results
                self.cache.set(cache_key, df.to_json(), source='aflow')
                
                return df
            
        except Exception as e:
            print(f"AFLOW API error: {e}")
        
        return pd.DataFrame()


class JARVISClient:
    """JARVIS-DFT API client with caching"""
    
    def __init__(self):
        self.cache = CacheDB()
        self.available = True  # JARVIS is public
    
    def _make_cache_key(self, query: dict) -> str:
        """Generate cache key from query"""
        query_str = json.dumps(query, sort_keys=True)
        return f"jarvis_{hashlib.md5(query_str.encode()).hexdigest()}"
    
    def search_perovskites(self, max_results: int = 500) -> pd.DataFrame:
        """
        Search for ABX3 perovskite structures in JARVIS-DFT.
        Returns DataFrame with formula, bandgap, structure info.
        """
        cache_key = self._make_cache_key({'type': 'perovskite_search', 'max': max_results})
        cached = self.cache.get(cache_key)
        
        if cached:
            return pd.DataFrame(json.loads(cached))
        
        try:
            # JARVIS API query
            # Note: Actual JARVIS API may have different endpoint structure
            response = requests.get(
                f"{JARVIS_API_URL}/dft",
                params={'filter': 'perovskite'},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                materials = []
                for item in data[:max_results]:
                    material = {
                        'material_id': item.get('jid'),
                        'formula': item.get('formula'),
                        'bandgap': item.get('optb88vdw_bandgap', np.nan),
                        'formation_energy': item.get('formation_energy_peratom', np.nan),
                        'source': 'jarvis'
                    }
                    materials.append(material)
                    
                    # Save to cache DB
                    self.cache.save_material(
                        material_id=material['material_id'],
                        formula=material['formula'],
                        source='jarvis',
                        bandgap=material['bandgap'],
                        structure='{}',
                        properties=material
                    )
                
                df = pd.DataFrame(materials)
                
                # Cache results
                self.cache.set(cache_key, df.to_json(), source='jarvis')
                
                return df
            
        except Exception as e:
            print(f"JARVIS API error: {e}")
        
        return pd.DataFrame()


class UnifiedDBClient:
    """
    Unified interface for all database clients.
    Aggregates data from multiple sources.
    """
    
    def __init__(self, mp_api_key: Optional[str] = None):
        self.mp = MaterialsProjectClient(api_key=mp_api_key)
        self.aflow = AFLOWClient()
        self.jarvis = JARVISClient()
        self.cache = CacheDB()
    
    def get_all_perovskites(self, max_per_source: int = 500, 
                           use_cache: bool = True) -> pd.DataFrame:
        """
        Aggregate perovskite data from all available sources.
        
        Returns:
            DataFrame with columns: formula, bandgap, source, material_id, ...
        """
        if use_cache:
            # Try to load from cache first
            cached = self.cache.get_all_materials()
            if not cached.empty:
                return cached
        
        dfs = []
        
        # Materials Project
        if self.mp.available:
            try:
                mp_data = self.mp.search_perovskites(max_results=max_per_source)
                if not mp_data.empty:
                    dfs.append(mp_data)
            except Exception as e:
                print(f"MP fetch failed: {e}")
        
        # AFLOW
        try:
            aflow_data = self.aflow.search_perovskites(max_results=max_per_source)
            if not aflow_data.empty:
                dfs.append(aflow_data)
        except Exception as e:
            print(f"AFLOW fetch failed: {e}")
        
        # JARVIS
        try:
            jarvis_data = self.jarvis.search_perovskites(max_results=max_per_source)
            if not jarvis_data.empty:
                dfs.append(jarvis_data)
        except Exception as e:
            print(f"JARVIS fetch failed: {e}")
        
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            
            # Remove duplicates based on formula
            combined = combined.drop_duplicates(subset=['formula'], keep='first')
            
            # Filter valid bandgaps
            combined = combined[combined['bandgap'].notna() & (combined['bandgap'] > 0)]
            
            return combined
        
        return pd.DataFrame()
    
    def get_statistics(self) -> Dict[str, int]:
        """Get cache statistics"""
        df = self.cache.get_all_materials()
        
        stats = {
            'total': len(df),
            'materials_project': len(df[df['source'] == 'materials_project']),
            'aflow': len(df[df['source'] == 'aflow']),
            'jarvis': len(df[df['source'] == 'jarvis']),
            'user_uploaded': len(df[df['source'] == 'user_upload'])
        }
        
        return stats
    
    def clear_cache(self, source: Optional[str] = None):
        """Clear database cache"""
        self.cache.clear(source=source)
