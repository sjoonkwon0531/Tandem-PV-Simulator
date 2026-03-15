"""
User Data Upload Parser
========================

Parse CSV/Excel files with experimental perovskite data.
Auto-detect columns, clean data, and merge with DB data.

Supported formats:
- CSV, Excel (.xlsx, .xls)
- Required columns: at least one of [formula, composition] and [bandgap, Eg, band_gap]
- Optional: Voc, Jsc, FF, PCE, PL_peak, stability, etc.

Author: OpenClaw Agent
Date: 2026-03-15
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import io


class UserDataParser:
    """Parse and validate user-uploaded experimental data"""
    
    # Column name mappings (case-insensitive)
    COLUMN_MAPPINGS = {
        'formula': ['formula', 'composition', 'compound', 'material', 'name'],
        'bandgap': ['bandgap', 'eg', 'band_gap', 'gap', 'e_g', 'optical_gap'],
        'voc': ['voc', 'v_oc', 'open_circuit_voltage', 'voltage'],
        'jsc': ['jsc', 'j_sc', 'short_circuit_current', 'current'],
        'ff': ['ff', 'fill_factor', 'fillfactor'],
        'pce': ['pce', 'efficiency', 'eta', 'power_conversion_efficiency'],
        'pl_peak': ['pl_peak', 'pl', 'photoluminescence', 'pl_wavelength'],
        'stability': ['stability', 't80', 't_80', 'lifetime'],
        'thickness': ['thickness', 'd', 'film_thickness'],
        'method': ['method', 'deposition', 'fabrication'],
        'notes': ['notes', 'comments', 'remarks']
    }
    
    def __init__(self):
        self.raw_data = None
        self.parsed_data = None
        self.validation_errors = []
    
    def parse(self, file_content: bytes, filename: str) -> pd.DataFrame:
        """
        Parse uploaded file and return cleaned DataFrame.
        
        Args:
            file_content: File bytes
            filename: Original filename (for extension detection)
        
        Returns:
            Parsed DataFrame
        """
        self.validation_errors = []
        
        # Detect file type
        ext = Path(filename).suffix.lower()
        
        try:
            if ext == '.csv':
                self.raw_data = pd.read_csv(io.BytesIO(file_content))
            elif ext in ['.xlsx', '.xls']:
                self.raw_data = pd.read_excel(io.BytesIO(file_content))
            else:
                raise ValueError(f"Unsupported file type: {ext}")
        
        except Exception as e:
            self.validation_errors.append(f"File read error: {e}")
            return pd.DataFrame()
        
        # Clean and standardize columns
        self.parsed_data = self._standardize_columns(self.raw_data)
        
        # Validate required columns
        if not self._validate_required_columns():
            return pd.DataFrame()
        
        # Clean data
        self.parsed_data = self._clean_data(self.parsed_data)
        
        # Parse formulas
        self.parsed_data = self._parse_formulas(self.parsed_data)
        
        return self.parsed_data
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map user column names to standard names"""
        df_copy = df.copy()
        column_map = {}
        
        for col in df_copy.columns:
            col_lower = col.lower().strip()
            
            # Check each standard column
            for standard, variations in self.COLUMN_MAPPINGS.items():
                if col_lower in variations:
                    column_map[col] = standard
                    break
        
        # Rename matched columns
        df_copy = df_copy.rename(columns=column_map)
        
        return df_copy
    
    def _validate_required_columns(self) -> bool:
        """Check that required columns are present"""
        required = ['formula', 'bandgap']
        
        missing = [col for col in required if col not in self.parsed_data.columns]
        
        if missing:
            self.validation_errors.append(
                f"Missing required columns: {missing}. "
                f"Upload must have 'formula' and 'bandgap' (or similar) columns."
            )
            return False
        
        return True
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data values"""
        df_copy = df.copy()
        
        # Remove empty rows
        df_copy = df_copy.dropna(how='all')
        
        # Clean bandgap column
        if 'bandgap' in df_copy.columns:
            # Convert to numeric, coerce errors to NaN
            df_copy['bandgap'] = pd.to_numeric(df_copy['bandgap'], errors='coerce')
            
            # Remove invalid bandgaps
            invalid_mask = (df_copy['bandgap'] <= 0) | (df_copy['bandgap'] > 10)
            n_invalid = invalid_mask.sum()
            
            if n_invalid > 0:
                self.validation_errors.append(
                    f"Warning: {n_invalid} rows with invalid bandgap (≤0 or >10 eV) removed."
                )
                df_copy = df_copy[~invalid_mask]
        
        # Clean formula column
        if 'formula' in df_copy.columns:
            # Remove whitespace
            df_copy['formula'] = df_copy['formula'].astype(str).str.strip()
            
            # Remove empty formulas
            df_copy = df_copy[df_copy['formula'] != '']
        
        # Clean numeric columns
        for col in ['voc', 'jsc', 'ff', 'pce', 'thickness']:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        
        return df_copy
    
    def _parse_formulas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse chemical formulas to extract composition info.
        Supports both standard (MAPbI3) and LaTeX (MAPbI₃) notation.
        """
        df_copy = df.copy()
        
        if 'formula' not in df_copy.columns:
            return df_copy
        
        compositions = []
        
        for formula in df_copy['formula']:
            comp = self._extract_composition(formula)
            compositions.append(comp)
        
        # Add composition info as new columns
        comp_df = pd.DataFrame(compositions)
        df_copy = pd.concat([df_copy, comp_df], axis=1)
        
        return df_copy
    
    def _extract_composition(self, formula: str) -> Dict[str, str]:
        """
        Extract A-site, B-site, X-site from ABX3 formula.
        
        Examples:
            MAPbI3 → A=MA, B=Pb, X=I
            FA0.87Cs0.13Pb(I0.62Br0.38)3 → A=FA0.87Cs0.13, B=Pb, X=I0.62Br0.38
        """
        # Normalize subscripts (₃ → 3)
        formula = formula.replace('₀', '0').replace('₁', '1').replace('₂', '2')
        formula = formula.replace('₃', '3').replace('₄', '4').replace('₅', '5')
        formula = formula.replace('₆', '6').replace('₇', '7').replace('₈', '8')
        formula = formula.replace('₉', '9')
        
        comp = {'A_site': '', 'B_site': '', 'X_site': ''}
        
        # Common A-site cations
        a_sites = ['MA', 'FA', 'Cs', 'Rb', 'K', 'Gua']
        # B-site metals
        b_sites = ['Pb', 'Sn', 'Ge', 'Sr', 'Ca', 'Mn']
        # X-site halides
        x_sites = ['I', 'Br', 'Cl', 'F']
        
        # Try to identify B-site first (usually unique)
        for b in b_sites:
            if b in formula:
                comp['B_site'] = b
                break
        
        # Try to identify A-site
        for a in a_sites:
            if a in formula:
                # Extract with stoichiometry if present
                match = re.search(f'({a}[0-9.]*)', formula)
                if match:
                    comp['A_site'] = match.group(1) if comp['A_site'] == '' else comp['A_site'] + match.group(1)
        
        # Try to identify X-site (halides)
        # Look for pattern like (I0.62Br0.38) or I3 or Br3
        halide_pattern = r'\(([IBrClF0-9.]+)\)|([IBrClF]+)3'
        match = re.search(halide_pattern, formula)
        
        if match:
            comp['X_site'] = match.group(1) if match.group(1) else match.group(2)
        
        return comp
    
    def get_validation_errors(self) -> List[str]:
        """Return list of validation errors/warnings"""
        return self.validation_errors
    
    def merge_with_db(self, df_user: pd.DataFrame, df_db: pd.DataFrame) -> pd.DataFrame:
        """
        Merge user data with database data.
        Adds 'source' column to distinguish.
        """
        # Add source column
        df_user_copy = df_user.copy()
        df_user_copy['source'] = 'user_upload'
        
        df_db_copy = df_db.copy()
        if 'source' not in df_db_copy.columns:
            df_db_copy['source'] = 'database'
        
        # Combine
        combined = pd.concat([df_db_copy, df_user_copy], ignore_index=True, sort=False)
        
        # Remove duplicates (prefer user data)
        combined = combined.drop_duplicates(subset=['formula'], keep='last')
        
        return combined
    
    def get_summary(self) -> Dict:
        """Get summary statistics of parsed data"""
        if self.parsed_data is None or self.parsed_data.empty:
            return {}
        
        summary = {
            'n_materials': len(self.parsed_data),
            'bandgap_range': (
                self.parsed_data['bandgap'].min(), 
                self.parsed_data['bandgap'].max()
            ) if 'bandgap' in self.parsed_data.columns else (None, None),
            'has_device_data': any(col in self.parsed_data.columns for col in ['voc', 'jsc', 'pce']),
            'columns': list(self.parsed_data.columns)
        }
        
        return summary


def example_csv() -> str:
    """Return example CSV template for user reference"""
    return """formula,bandgap,voc,jsc,ff,pce,notes
MAPbI3,1.59,1.12,23.4,0.81,21.3,Standard reference
FAPbI3,1.51,1.08,24.2,0.79,20.6,Alpha phase
CsPbI3,1.72,1.15,19.8,0.75,17.1,Quantum dots
FA0.85MA0.15PbI3,1.55,1.10,23.8,0.82,21.5,Mixed cation
Cs0.05FA0.95PbI3,1.53,1.09,24.0,0.80,20.9,Cs-doped FA
"""


def example_excel_description() -> str:
    """Return description of Excel format"""
    return """
**Excel Format Requirements:**

**Required columns:**
- `formula` or `composition`: Chemical formula (e.g., MAPbI3, FA0.87Cs0.13PbI3)
- `bandgap` or `Eg`: Bandgap in eV

**Optional columns:**
- `voc`, `jsc`, `ff`, `pce`: Device performance metrics
- `pl_peak`: Photoluminescence peak (nm or eV)
- `stability`: T80 lifetime (hours)
- `thickness`: Film thickness (nm)
- `method`: Deposition method
- `notes`: Any comments

**Example row:**
```
formula: FA0.87Cs0.13Pb(I0.62Br0.38)3
bandgap: 1.68
voc: 1.27
jsc: 21.8
ff: 0.83
pce: 23.1
```
"""
