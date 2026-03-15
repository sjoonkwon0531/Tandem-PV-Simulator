"""
Session Persistence for V5
===========================

Save/load user sessions including:
- Uploaded data
- Model state
- Bayesian optimization history
- Multi-objective preferences
- Experiment queue

Format: JSON-based session files

Author: OpenClaw Agent
Date: 2026-03-15 (V5)
"""

import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime


class SessionManager:
    """
    Manage user sessions for personalized learning.
    
    A session captures:
    - Uploaded experimental data
    - Trained model state
    - BO history and suggestions
    - Multi-objective weights
    - Experiment planner queue
    """
    
    def __init__(self, session_dir: str = "./sessions"):
        """
        Args:
            session_dir: Directory to store session files
        """
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
    
    def save_session(self, session_data: Dict[str, Any], 
                    session_name: str) -> str:
        """
        Save complete session to disk.
        
        Args:
            session_data: Dictionary containing all session state
            session_name: Human-readable session name
        
        Returns:
            Path to saved session file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"{session_name}_{timestamp}"
        
        session_path = self.session_dir / session_id
        session_path.mkdir(exist_ok=True)
        
        # Save metadata
        metadata = {
            'session_id': session_id,
            'session_name': session_name,
            'created_at': datetime.now().isoformat(),
            'version': 'v5.0',
            'description': session_data.get('description', '')
        }
        
        with open(session_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save user data (CSV)
        if 'user_data' in session_data and session_data['user_data'] is not None:
            df = session_data['user_data']
            df.to_csv(session_path / 'user_data.csv', index=False)
        
        # Save model state (joblib)
        if 'ml_model' in session_data and session_data['ml_model'] is not None:
            model_path = session_path / 'ml_model.joblib'
            joblib.dump(session_data['ml_model'], model_path)
        
        # Save BO state
        if 'bo_state' in session_data:
            bo_state = session_data['bo_state']
            
            # BO model (GP)
            if 'bo_optimizer' in bo_state and bo_state['bo_optimizer'] is not None:
                joblib.dump(bo_state['bo_optimizer'], session_path / 'bo_optimizer.joblib')
            
            # BO history (DataFrame)
            if 'bo_history' in bo_state and bo_state['bo_history'] is not None:
                bo_state['bo_history'].to_csv(session_path / 'bo_history.csv', index=False)
            
            # BO config
            bo_config = {
                'target_bandgap': bo_state.get('target_bandgap'),
                'acq_function': bo_state.get('acq_function'),
                'n_iterations': bo_state.get('n_iterations', 0)
            }
            
            with open(session_path / 'bo_config.json', 'w') as f:
                json.dump(bo_config, f, indent=2)
        
        # Save multi-objective preferences
        if 'mo_weights' in session_data:
            with open(session_path / 'mo_weights.json', 'w') as f:
                json.dump(session_data['mo_weights'], f, indent=2)
        
        # Save experiment queue
        if 'experiment_queue' in session_data and session_data['experiment_queue'] is not None:
            session_data['experiment_queue'].to_csv(session_path / 'experiment_queue.csv', index=False)
        
        # Save training history (if fine-tuning performed)
        if 'training_history' in session_data:
            with open(session_path / 'training_history.json', 'w') as f:
                # Convert numpy types to native Python
                history = self._serialize_training_history(session_data['training_history'])
                json.dump(history, f, indent=2)
        
        return str(session_path)
    
    def load_session(self, session_id: str) -> Dict[str, Any]:
        """
        Load session from disk.
        
        Args:
            session_id: Session ID or name
        
        Returns:
            Dictionary with session data
        """
        # Find session directory
        session_path = self._find_session_path(session_id)
        
        if session_path is None:
            raise FileNotFoundError(f"Session '{session_id}' not found")
        
        session_data = {}
        
        # Load metadata
        with open(session_path / 'metadata.json', 'r') as f:
            session_data['metadata'] = json.load(f)
        
        # Load user data
        user_data_path = session_path / 'user_data.csv'
        if user_data_path.exists():
            session_data['user_data'] = pd.read_csv(user_data_path)
        
        # Load model
        model_path = session_path / 'ml_model.joblib'
        if model_path.exists():
            session_data['ml_model'] = joblib.load(model_path)
        
        # Load BO state
        bo_state = {}
        
        bo_optimizer_path = session_path / 'bo_optimizer.joblib'
        if bo_optimizer_path.exists():
            bo_state['bo_optimizer'] = joblib.load(bo_optimizer_path)
        
        bo_history_path = session_path / 'bo_history.csv'
        if bo_history_path.exists():
            bo_state['bo_history'] = pd.read_csv(bo_history_path)
        
        bo_config_path = session_path / 'bo_config.json'
        if bo_config_path.exists():
            with open(bo_config_path, 'r') as f:
                bo_state.update(json.load(f))
        
        if bo_state:
            session_data['bo_state'] = bo_state
        
        # Load multi-objective weights
        mo_weights_path = session_path / 'mo_weights.json'
        if mo_weights_path.exists():
            with open(mo_weights_path, 'r') as f:
                session_data['mo_weights'] = json.load(f)
        
        # Load experiment queue
        queue_path = session_path / 'experiment_queue.csv'
        if queue_path.exists():
            session_data['experiment_queue'] = pd.read_csv(queue_path)
        
        # Load training history
        history_path = session_path / 'training_history.json'
        if history_path.exists():
            with open(history_path, 'r') as f:
                session_data['training_history'] = json.load(f)
        
        return session_data
    
    def list_sessions(self) -> pd.DataFrame:
        """
        List all saved sessions.
        
        Returns:
            DataFrame with session info
        """
        sessions = []
        
        for session_path in self.session_dir.iterdir():
            if session_path.is_dir():
                metadata_path = session_path / 'metadata.json'
                
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Get file sizes
                    total_size = sum(f.stat().st_size for f in session_path.rglob('*') if f.is_file())
                    
                    sessions.append({
                        'session_id': metadata['session_id'],
                        'session_name': metadata['session_name'],
                        'created_at': metadata['created_at'],
                        'version': metadata.get('version', 'unknown'),
                        'size_mb': total_size / (1024 * 1024),
                        'path': str(session_path)
                    })
        
        if sessions:
            df = pd.DataFrame(sessions)
            df = df.sort_values('created_at', ascending=False)
            return df
        else:
            return pd.DataFrame()
    
    def delete_session(self, session_id: str):
        """
        Delete a session.
        
        Args:
            session_id: Session ID to delete
        """
        session_path = self._find_session_path(session_id)
        
        if session_path is None:
            raise FileNotFoundError(f"Session '{session_id}' not found")
        
        # Delete directory
        import shutil
        shutil.rmtree(session_path)
    
    def _find_session_path(self, session_id: str) -> Optional[Path]:
        """Find session path by ID or name"""
        # Try exact ID match first
        for session_path in self.session_dir.iterdir():
            if session_path.is_dir() and session_path.name == session_id:
                return session_path
        
        # Try name match
        for session_path in self.session_dir.iterdir():
            if session_path.is_dir():
                metadata_path = session_path / 'metadata.json'
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    if metadata.get('session_name') == session_id:
                        return session_path
        
        return None
    
    def _serialize_training_history(self, history: List[Dict]) -> List[Dict]:
        """Convert numpy types to native Python for JSON serialization"""
        serialized = []
        
        for entry in history:
            serialized_entry = {}
            for key, value in entry.items():
                if isinstance(value, (np.integer, np.floating)):
                    serialized_entry[key] = float(value)
                elif isinstance(value, np.ndarray):
                    serialized_entry[key] = value.tolist()
                else:
                    serialized_entry[key] = value
            
            serialized.append(serialized_entry)
        
        return serialized
    
    def export_session_report(self, session_id: str, output_path: str):
        """
        Export session summary as PDF/HTML report.
        
        Args:
            session_id: Session to export
            output_path: Output file path (.html or .pdf)
        """
        session_data = self.load_session(session_id)
        
        # Generate HTML report
        html = self._generate_html_report(session_data)
        
        # Write HTML
        if output_path.endswith('.html'):
            with open(output_path, 'w') as f:
                f.write(html)
        elif output_path.endswith('.pdf'):
            # Convert to PDF (requires weasyprint or similar)
            try:
                from weasyprint import HTML
                HTML(string=html).write_pdf(output_path)
            except ImportError:
                # Fallback to HTML if weasyprint not available
                html_path = output_path.replace('.pdf', '.html')
                with open(html_path, 'w') as f:
                    f.write(html)
                print(f"PDF export requires weasyprint. Saved as HTML instead: {html_path}")
    
    def _generate_html_report(self, session_data: Dict) -> str:
        """Generate HTML report from session data"""
        metadata = session_data.get('metadata', {})
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Session Report: {metadata.get('session_name', 'Unknown')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #667eea; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #667eea; color: white; }}
            </style>
        </head>
        <body>
            <h1>AlphaMaterials V5 Session Report</h1>
            <h2>{metadata.get('session_name', 'Unknown Session')}</h2>
            <p><strong>Created:</strong> {metadata.get('created_at', 'Unknown')}</p>
            <p><strong>Version:</strong> {metadata.get('version', 'Unknown')}</p>
            
            <h3>User Data</h3>
        """
        
        if 'user_data' in session_data:
            df = session_data['user_data']
            html += f"<p>{len(df)} materials uploaded</p>"
            html += df.head(10).to_html()
        else:
            html += "<p>No user data</p>"
        
        html += """
        </body>
        </html>
        """
        
        return html


def create_default_session() -> Dict[str, Any]:
    """Create empty session template"""
    return {
        'description': '',
        'user_data': None,
        'ml_model': None,
        'bo_state': {
            'bo_optimizer': None,
            'bo_history': None,
            'target_bandgap': 1.68,
            'acq_function': 'ei',
            'n_iterations': 0
        },
        'mo_weights': {
            'obj_bandgap_match': 0.4,
            'obj_stability': 0.3,
            'obj_synthesizability': 0.2,
            'obj_cost': 0.1
        },
        'experiment_queue': None,
        'training_history': []
    }
