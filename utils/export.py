"""
Publication-Ready Export Tools
===============================

Generate formatted tables, figures, and text for scientific publications.

Features:
- LaTeX tables (booktabs format)
- CSV export
- High-DPI figures (SVG, PNG @ 300 DPI)
- Auto-generated methods section text
- BibTeX references for models/databases used

Author: OpenClaw Agent
Date: 2026-03-15 (V6)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.io as pio


class PublicationExporter:
    """
    Export discovery campaign results in publication-ready format.
    """
    
    # BibTeX references for citations
    BIBTEX_REFS = {
        'materials_project': """@article{Jain2013,
  author = {Jain, Anubhav and Ong, Shyue Ping and Hautier, Geoffroy and Chen, Wei and Richards, William Davidson and Dacek, Stephen and Cholia, Shreyas and Gunter, Dan and Skinner, David and Ceder, Gerbrand and Persson, Kristin A.},
  title = {Commentary: The Materials Project: A materials genome approach to accelerating materials innovation},
  journal = {APL Materials},
  volume = {1},
  number = {1},
  pages = {011002},
  year = {2013},
  doi = {10.1063/1.4812323}
}""",
        'xgboost': """@inproceedings{Chen2016,
  author = {Chen, Tianqi and Guestrin, Carlos},
  title = {XGBoost: A Scalable Tree Boosting System},
  booktitle = {Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  year = {2016},
  pages = {785--794},
  doi = {10.1145/2939672.2939785}
}""",
        'sklearn': """@article{Pedregosa2011,
  author = {Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V. and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P. and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
  title = {Scikit-learn: Machine Learning in Python},
  journal = {Journal of Machine Learning Research},
  volume = {12},
  pages = {2825--2830},
  year = {2011}
}""",
        'gaussian_process': """@book{Rasmussen2006,
  author = {Rasmussen, Carl Edward and Williams, Christopher K. I.},
  title = {Gaussian Processes for Machine Learning},
  publisher = {MIT Press},
  year = {2006},
  isbn = {026218253X}
}""",
        'bayesian_optimization': """@article{Shahriari2016,
  author = {Shahriari, Bobak and Swersky, Kevin and Wang, Ziyu and Adams, Ryan P. and de Freitas, Nando},
  title = {Taking the Human Out of the Loop: A Review of Bayesian Optimization},
  journal = {Proceedings of the IEEE},
  volume = {104},
  number = {1},
  pages = {148--175},
  year = {2016},
  doi = {10.1109/JPROC.2015.2494218}
}"""
    }
    
    def __init__(self, output_dir: Path = None):
        """
        Args:
            output_dir: Directory for exported files
        """
        self.output_dir = output_dir or Path('./exports')
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def export_table_latex(self, df: pd.DataFrame, 
                          caption: str,
                          label: str,
                          filename: str = None) -> str:
        """
        Export DataFrame as LaTeX table (booktabs format).
        
        Args:
            df: DataFrame to export
            caption: Table caption
            label: LaTeX label (e.g., 'tab:candidates')
            filename: Output filename (optional, will also return string)
        
        Returns:
            LaTeX table string
        """
        # Round numeric columns
        df_export = df.copy()
        for col in df_export.columns:
            if df_export[col].dtype in [np.float64, np.float32]:
                df_export[col] = df_export[col].round(3)
        
        # Generate LaTeX
        latex_str = df_export.to_latex(
            index=False,
            escape=False,
            column_format='l' + 'c' * (len(df.columns) - 1),
            caption=caption,
            label=label
        )
        
        # Add booktabs styling
        latex_str = latex_str.replace('\\toprule', '\\toprule\n')
        latex_str = latex_str.replace('\\midrule', '\\midrule\n')
        latex_str = latex_str.replace('\\bottomrule', '\\bottomrule')
        
        # Save to file
        if filename:
            filepath = self.output_dir / filename
            with open(filepath, 'w') as f:
                f.write(latex_str)
        
        return latex_str
    
    def export_table_csv(self, df: pd.DataFrame, filename: str) -> Path:
        """
        Export DataFrame as CSV.
        
        Returns:
            Path to saved file
        """
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        return filepath
    
    def export_figure_svg(self, fig: go.Figure, filename: str, 
                         width: int = 800, height: int = 600) -> Path:
        """
        Export Plotly figure as SVG (vector, publication-quality).
        
        Args:
            fig: Plotly figure
            filename: Output filename (without extension)
            width, height: Figure dimensions (pixels)
        
        Returns:
            Path to saved file
        """
        filepath = self.output_dir / f"{filename}.svg"
        
        fig.update_layout(
            width=width,
            height=height,
            font=dict(family='Arial', size=12)
        )
        
        pio.write_image(fig, str(filepath), format='svg')
        return filepath
    
    def export_figure_png(self, fig: go.Figure, filename: str,
                         width: int = 800, height: int = 600,
                         dpi: int = 300) -> Path:
        """
        Export Plotly figure as high-DPI PNG.
        
        Args:
            fig: Plotly figure
            filename: Output filename (without extension)
            width, height: Figure dimensions (pixels at 300 DPI)
            dpi: Dots per inch (300 for print quality)
        
        Returns:
            Path to saved file
        """
        filepath = self.output_dir / f"{filename}.png"
        
        # Scale dimensions for DPI
        scale = dpi / 96  # 96 DPI is default screen resolution
        
        fig.update_layout(
            width=width,
            height=height,
            font=dict(family='Arial', size=12)
        )
        
        pio.write_image(
            fig, 
            str(filepath), 
            format='png',
            scale=scale
        )
        return filepath
    
    def generate_methods_section(self, 
                                 used_databases: List[str] = None,
                                 used_ml_models: List[str] = None,
                                 used_bo: bool = False,
                                 used_mo: bool = False,
                                 n_experiments: int = 0,
                                 custom_text: str = None) -> str:
        """
        Auto-generate Methods section text for paper.
        
        Args:
            used_databases: List of databases used (e.g., ['Materials Project', 'AFLOW'])
            used_ml_models: List of ML models (e.g., ['XGBoost', 'Gaussian Process'])
            used_bo: Whether Bayesian optimization was used
            used_mo: Whether multi-objective optimization was used
            n_experiments: Number of experimental data points
            custom_text: Additional custom text to append
        
        Returns:
            Formatted methods text
        """
        methods = []
        
        # Header
        methods.append("## Computational Methods\n")
        
        # Database section
        if used_databases:
            db_text = "### Data Sources\n\n"
            db_text += "We compiled perovskite property data from the following databases: "
            db_text += ", ".join(used_databases) + ". "
            db_text += "DFT-calculated bandgaps were extracted via API queries and cached locally for reproducibility.\n"
            methods.append(db_text)
        
        # ML section
        if used_ml_models:
            ml_text = "\n### Machine Learning Surrogate Models\n\n"
            ml_text += "Composition-to-property mappings were modeled using supervised learning. "
            ml_text += f"We employed {', '.join(used_ml_models)} "
            ml_text += "trained on the combined DFT database and experimental measurements. "
            ml_text += "Compositions were featurized using elemental properties (ionic radius, electronegativity, valence) "
            ml_text += "and structural descriptors (tolerance factor, octahedral factor, mixing entropy). "
            ml_text += f"The model was trained on {n_experiments} experimental data points "
            ml_text += "with 5-fold cross-validation for uncertainty quantification.\n"
            methods.append(ml_text)
        
        # BO section
        if used_bo:
            bo_text = "\n### Bayesian Optimization\n\n"
            bo_text += "Next-experiment selection was guided by Bayesian optimization using a Gaussian Process (GP) surrogate. "
            bo_text += "The GP was fitted on experimental data with a Matérn kernel (ν=2.5) and noise variance α=0.01. "
            bo_text += "Candidate compositions were ranked using the Expected Improvement (EI) acquisition function, "
            bo_text += "which balances exploitation (high predicted performance) with exploration (high uncertainty). "
            bo_text += "This active learning loop accelerates discovery by focusing synthesis efforts on high-value experiments.\n"
            methods.append(bo_text)
        
        # MO section
        if used_mo:
            mo_text = "\n### Multi-Objective Optimization\n\n"
            mo_text += "We simultaneously optimized four objectives: (1) bandgap matching to target value, "
            mo_text += "(2) thermodynamic stability (tolerance factor proximity to 0.95), "
            mo_text += "(3) synthesizability (low mixing entropy), and (4) raw material cost. "
            mo_text += "The Pareto front was calculated to identify compositions that are non-dominated across objectives. "
            mo_text += "Final recommendations were obtained via weighted scalarization with user-defined priorities.\n"
            methods.append(mo_text)
        
        # Software
        software_text = "\n### Software and Reproducibility\n\n"
        software_text += "All analyses were performed using Python 3.11 with scikit-learn (v1.3), XGBoost (v2.0), "
        software_text += "and SciPy (v1.11). Code and trained models are available at [GitHub repository]. "
        software_text += "DFT data is cached in SQLite format for offline reproducibility.\n"
        methods.append(software_text)
        
        # Custom text
        if custom_text:
            methods.append(f"\n{custom_text}\n")
        
        return "\n".join(methods)
    
    def generate_bibtex_file(self, 
                            used_tools: List[str] = None,
                            filename: str = 'references.bib') -> Path:
        """
        Generate BibTeX file with relevant citations.
        
        Args:
            used_tools: List of tools used (keys from BIBTEX_REFS)
            filename: Output filename
        
        Returns:
            Path to .bib file
        """
        if used_tools is None:
            used_tools = list(self.BIBTEX_REFS.keys())
        
        bib_entries = []
        
        for tool in used_tools:
            if tool in self.BIBTEX_REFS:
                bib_entries.append(self.BIBTEX_REFS[tool])
        
        bib_text = "\n\n".join(bib_entries)
        
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            f.write(bib_text)
        
        return filepath
    
    def export_campaign_summary(self, 
                               campaign_data: Dict,
                               filename: str = 'campaign_summary.json') -> Path:
        """
        Export full campaign summary as JSON.
        
        Args:
            campaign_data: Dict with campaign metadata
            filename: Output filename
        
        Returns:
            Path to JSON file
        """
        # Add timestamp
        campaign_data['exported_at'] = datetime.now().isoformat()
        campaign_data['version'] = 'AlphaMaterials V6'
        
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(campaign_data, f, indent=2, default=str)
        
        return filepath
    
    def create_supplementary_package(self,
                                    candidates_df: pd.DataFrame,
                                    pareto_df: pd.DataFrame,
                                    cost_analysis_df: pd.DataFrame,
                                    figures: Dict[str, go.Figure],
                                    methods_text: str) -> Path:
        """
        Create complete supplementary materials package.
        
        Creates:
        - SI_Table_S1_candidates.csv
        - SI_Table_S2_pareto.csv
        - SI_Table_S3_cost.csv
        - SI_Figure_*.png (all figures @ 300 DPI)
        - SI_Methods.txt
        - references.bib
        
        Args:
            candidates_df: Top candidates from inverse design
            pareto_df: Pareto-optimal materials
            cost_analysis_df: Cost comparison table
            figures: Dict of {name: plotly_figure}
            methods_text: Methods section text
        
        Returns:
            Path to output directory
        """
        # Create SI subdirectory
        si_dir = self.output_dir / 'supplementary_information'
        si_dir.mkdir(exist_ok=True)
        
        # Tables
        candidates_df.to_csv(si_dir / 'SI_Table_S1_candidates.csv', index=False)
        pareto_df.to_csv(si_dir / 'SI_Table_S2_pareto.csv', index=False)
        cost_analysis_df.to_csv(si_dir / 'SI_Table_S3_cost_analysis.csv', index=False)
        
        # Figures
        for i, (name, fig) in enumerate(figures.items(), 1):
            filepath = si_dir / f'SI_Figure_S{i}_{name}.png'
            pio.write_image(fig, str(filepath), format='png', scale=300/96, width=800, height=600)
        
        # Methods
        with open(si_dir / 'SI_Methods.txt', 'w') as f:
            f.write(methods_text)
        
        # BibTeX
        self.output_dir = si_dir  # Temporarily redirect
        self.generate_bibtex_file()
        self.output_dir = si_dir.parent  # Restore
        
        # README
        readme = f"""# Supplementary Information
        
## AlphaMaterials V6: Generative Inverse Design for Perovskite Solar Cells

**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Contents

**Tables:**
- `SI_Table_S1_candidates.csv` — Top candidates from inverse design ({len(candidates_df)} entries)
- `SI_Table_S2_pareto.csv` — Pareto-optimal materials ({len(pareto_df)} entries)
- `SI_Table_S3_cost_analysis.csv` — Techno-economic comparison

**Figures:**
{chr(10).join([f'- `SI_Figure_S{i+1}_{name}.png` — {name.replace("_", " ").title()}' for i, name in enumerate(figures.keys())])}

**Methods:**
- `SI_Methods.txt` — Computational methods section text
- `references.bib` — BibTeX citations

### Reproducibility

All data was generated using AlphaMaterials V6 (open source).
Repository: [GitHub link]
Session files: Available upon request

### Citation

[Full paper citation]

DOI: [...]
"""
        
        with open(si_dir / 'README.md', 'w') as f:
            f.write(readme)
        
        return si_dir


def format_property_table(df: pd.DataFrame, 
                         cols_to_include: List[str],
                         col_labels: Dict[str, str] = None,
                         significant_figures: int = 3) -> pd.DataFrame:
    """
    Format DataFrame for publication table.
    
    Args:
        df: Input DataFrame
        cols_to_include: Columns to include in output
        col_labels: Mapping of column names to publication labels
        significant_figures: Number of significant figures for floats
    
    Returns:
        Formatted DataFrame
    """
    df_out = df[cols_to_include].copy()
    
    # Rename columns
    if col_labels:
        df_out = df_out.rename(columns=col_labels)
    
    # Round numeric columns
    for col in df_out.columns:
        if df_out[col].dtype in [np.float64, np.float32]:
            df_out[col] = df_out[col].apply(lambda x: f"{x:.{significant_figures}g}")
    
    return df_out


def create_graphical_abstract(top_candidate: str,
                             predicted_bandgap: float,
                             cost_per_watt: float,
                             pareto_rank: int) -> go.Figure:
    """
    Create graphical abstract figure for paper.
    
    Shows:
    - Discovery workflow (DB → ML → BO → Experiment)
    - Top candidate properties
    - Key metrics
    
    Returns:
        Plotly figure suitable for graphical abstract
    """
    fig = go.Figure()
    
    # Text annotations for workflow
    workflow_steps = [
        "DFT Database<br>(500+ materials)",
        "ML Surrogate<br>(XGBoost + GP)",
        "Bayesian Opt<br>(Active Learning)",
        "Inverse Design<br>(Target → Candidates)",
        f"<b>{top_candidate}</b><br>Eg={predicted_bandgap:.2f} eV<br>${cost_per_watt:.2f}/W"
    ]
    
    x_positions = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for i, (step, x) in enumerate(zip(workflow_steps, x_positions)):
        # Box
        fig.add_shape(
            type='rect',
            x0=x-0.08, x1=x+0.08,
            y0=0.4, y1=0.6,
            fillcolor='lightblue' if i < 4 else 'lightgreen',
            line=dict(color='black', width=2)
        )
        
        # Text
        fig.add_annotation(
            x=x, y=0.5,
            text=step,
            showarrow=False,
            font=dict(size=10, color='black')
        )
        
        # Arrow
        if i < len(workflow_steps) - 1:
            fig.add_annotation(
                x=(x + x_positions[i+1]) / 2,
                y=0.5,
                text='→',
                showarrow=False,
                font=dict(size=24, color='gray')
            )
    
    fig.update_layout(
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=1000,
        height=300,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    return fig
