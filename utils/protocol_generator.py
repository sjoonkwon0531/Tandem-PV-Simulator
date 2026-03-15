#!/usr/bin/env python3
"""
Experiment Protocol Generator for AlphaMaterials V10
====================================================

Generate detailed synthesis protocols from AI-suggested compositions:
- Step-by-step procedures
- Precursor preparation
- Solution mixing
- Spin-coating parameters
- Annealing profiles
- Safety warnings
- Equipment list
- Time and cost estimates

Author: OpenClaw Agent
Date: 2026-03-15
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class ProtocolStep:
    """A single step in the synthesis protocol"""
    step_number: int
    action: str
    details: str
    duration: str
    equipment: List[str]
    safety_notes: str = ""
    critical: bool = False


@dataclass
class SynthesisProtocol:
    """Complete synthesis protocol"""
    composition: str
    title: str
    steps: List[ProtocolStep]
    total_time: str
    total_cost: float
    safety_warnings: List[str]
    equipment_list: List[str]
    precursors: Dict[str, Dict]
    notes: str = ""


class ProtocolGenerator:
    """Generate synthesis protocols for perovskite materials"""
    
    def __init__(self):
        # Precursor database
        self.precursors = {
            'MA': {
                'name': 'Methylammonium iodide (MAI)',
                'formula': 'CH3NH3I',
                'supplier': 'Sigma-Aldrich',
                'purity': '≥99%',
                'cost_per_g': 15.0,
                'hazards': ['Irritant'],
                'cas': '14965-49-2'
            },
            'FA': {
                'name': 'Formamidinium iodide (FAI)',
                'formula': 'CH(NH2)2I',
                'supplier': 'GreatCell Solar',
                'purity': '≥99.5%',
                'cost_per_g': 25.0,
                'hazards': ['Irritant'],
                'cas': '879643-71-7'
            },
            'Cs': {
                'name': 'Cesium iodide (CsI)',
                'formula': 'CsI',
                'supplier': 'Sigma-Aldrich',
                'purity': '99.999%',
                'cost_per_g': 8.0,
                'hazards': ['Irritant'],
                'cas': '7789-17-5'
            },
            'Pb': {
                'name': 'Lead(II) iodide (PbI2)',
                'formula': 'PbI2',
                'supplier': 'TCI Chemicals',
                'purity': '99.99%',
                'cost_per_g': 5.0,
                'hazards': ['Toxic', 'Reproductive hazard', 'Environmental hazard'],
                'cas': '10101-63-0',
                'special_handling': 'Use in fume hood. Wear gloves and safety glasses. Dispose as hazardous waste.'
            },
            'Sn': {
                'name': 'Tin(II) iodide (SnI2)',
                'formula': 'SnI2',
                'supplier': 'Alfa Aesar',
                'purity': '99.99%',
                'cost_per_g': 12.0,
                'hazards': ['Irritant', 'Air-sensitive'],
                'cas': '10294-70-9',
                'special_handling': 'Handle under inert atmosphere (N2 or Ar). Oxidizes rapidly in air.'
            },
            'Br': {
                'name': 'Lead(II) bromide (PbBr2)',
                'formula': 'PbBr2',
                'supplier': 'Sigma-Aldrich',
                'purity': '≥98%',
                'cost_per_g': 6.0,
                'hazards': ['Toxic', 'Reproductive hazard'],
                'cas': '10031-22-8'
            },
            'Cl': {
                'name': 'Lead(II) chloride (PbCl2)',
                'formula': 'PbCl2',
                'supplier': 'Sigma-Aldrich',
                'purity': '≥98%',
                'cost_per_g': 4.0,
                'hazards': ['Toxic', 'Reproductive hazard'],
                'cas': '7758-95-4'
            }
        }
        
        # Solvent database
        self.solvents = {
            'DMF': {
                'name': 'N,N-Dimethylformamide',
                'supplier': 'Sigma-Aldrich',
                'purity': 'anhydrous, 99.8%',
                'cost_per_L': 50.0,
                'hazards': ['Toxic', 'Reproductive hazard'],
                'bp': '153°C'
            },
            'DMSO': {
                'name': 'Dimethyl sulfoxide',
                'supplier': 'Sigma-Aldrich',
                'purity': 'anhydrous, ≥99.9%',
                'cost_per_L': 45.0,
                'hazards': ['Irritant', 'Combustible'],
                'bp': '189°C'
            },
            'GBL': {
                'name': 'γ-Butyrolactone',
                'supplier': 'Sigma-Aldrich',
                'purity': '≥99%',
                'cost_per_L': 55.0,
                'hazards': ['Harmful if swallowed'],
                'bp': '204°C'
            }
        }
        
        # Equipment database
        self.equipment = {
            'spin_coater': 'Spin coater (Laurell WS-650 or equivalent)',
            'hotplate': 'Hotplate with temperature control (±1°C)',
            'glovebox': 'N2-filled glovebox (O2, H2O < 1 ppm)',
            'balance': 'Analytical balance (0.1 mg precision)',
            'vials': 'Glass vials (20 mL, amber)',
            'pipettes': 'Micropipettes (10-100 µL, 100-1000 µL)',
            'filters': 'PTFE syringe filters (0.45 µm)',
            'substrates': 'FTO-coated glass (2×2 cm)',
            'vacuum': 'Vacuum oven',
            'fume_hood': 'Chemical fume hood'
        }
    
    def generate_protocol(self, composition: str, 
                         target_properties: Optional[Dict] = None) -> SynthesisProtocol:
        """
        Generate complete synthesis protocol for given composition
        
        Args:
            composition: Chemical formula (e.g., "MAPbI3", "Cs0.1FA0.9PbI2.8Br0.2")
            target_properties: Optional target properties dict
        
        Returns:
            SynthesisProtocol object
        """
        # Parse composition
        parsed = self._parse_composition(composition)
        
        # Determine synthesis method
        method = self._select_synthesis_method(parsed)
        
        # Build protocol steps
        steps = []
        
        if method == 'one_step':
            steps = self._generate_one_step_protocol(parsed)
        elif method == 'two_step':
            steps = self._generate_two_step_protocol(parsed)
        elif method == 'antisolvent':
            steps = self._generate_antisolvent_protocol(parsed)
        
        # Calculate time and cost
        total_time = self._calculate_total_time(steps)
        total_cost = self._calculate_total_cost(parsed)
        
        # Compile safety warnings
        safety_warnings = self._compile_safety_warnings(parsed)
        
        # Compile equipment list
        equipment_list = self._compile_equipment_list(steps)
        
        # Get precursor details
        precursor_details = self._get_precursor_details(parsed)
        
        protocol = SynthesisProtocol(
            composition=composition,
            title=f"Synthesis of {composition} Perovskite Thin Films",
            steps=steps,
            total_time=total_time,
            total_cost=total_cost,
            safety_warnings=safety_warnings,
            equipment_list=equipment_list,
            precursors=precursor_details,
            notes=self._generate_notes(parsed)
        )
        
        return protocol
    
    def _parse_composition(self, composition: str) -> Dict:
        """Parse composition string into elements and ratios"""
        # Simplified parsing - in production, use proper chemistry parser
        parsed = {
            'a_site': [],
            'b_site': [],
            'x_site': [],
            'mixed_halides': False,
            'lead_free': False
        }
        
        # Detect A-site cations
        if 'MA' in composition:
            parsed['a_site'].append(('MA', self._extract_ratio(composition, 'MA')))
        if 'FA' in composition:
            parsed['a_site'].append(('FA', self._extract_ratio(composition, 'FA')))
        if 'Cs' in composition:
            parsed['a_site'].append(('Cs', self._extract_ratio(composition, 'Cs')))
        
        # Detect B-site metal
        if 'Pb' in composition:
            parsed['b_site'].append(('Pb', 1.0))
        elif 'Sn' in composition:
            parsed['b_site'].append(('Sn', 1.0))
            parsed['lead_free'] = True
        
        # Detect X-site halides
        if 'I' in composition:
            parsed['x_site'].append(('I', self._extract_ratio(composition, 'I')))
        if 'Br' in composition:
            parsed['x_site'].append(('Br', self._extract_ratio(composition, 'Br')))
            parsed['mixed_halides'] = True
        if 'Cl' in composition:
            parsed['x_site'].append(('Cl', self._extract_ratio(composition, 'Cl')))
            parsed['mixed_halides'] = True
        
        return parsed
    
    def _extract_ratio(self, composition: str, element: str) -> float:
        """Extract stoichiometric ratio from composition string"""
        import re
        pattern = f'{element}([0-9.]*)'
        match = re.search(pattern, composition)
        if match and match.group(1):
            return float(match.group(1))
        return 1.0
    
    def _select_synthesis_method(self, parsed: Dict) -> str:
        """Select appropriate synthesis method"""
        if parsed['mixed_halides']:
            return 'one_step'  # One-step for mixed halides
        elif parsed['lead_free']:
            return 'antisolvent'  # Antisolvent for Sn-based (air-sensitive)
        else:
            return 'one_step'  # Default one-step
    
    def _generate_one_step_protocol(self, parsed: Dict) -> List[ProtocolStep]:
        """Generate one-step spin-coating protocol"""
        steps = []
        
        # Step 1: Precursor weighing
        steps.append(ProtocolStep(
            step_number=1,
            action="Weigh precursors",
            details=self._generate_weighing_details(parsed),
            duration="15 min",
            equipment=['balance', 'vials'],
            safety_notes="Wear gloves and safety glasses. Work in fume hood for lead compounds.",
            critical=True
        ))
        
        # Step 2: Dissolve in solvent
        steps.append(ProtocolStep(
            step_number=2,
            action="Dissolve in mixed solvent",
            details="Add DMF:DMSO (4:1 v/v) to reach 1.3 M concentration. Stir at 60°C for 2 hours until fully dissolved.",
            duration="2 hours",
            equipment=['hotplate', 'vials'],
            safety_notes="DMF and DMSO are toxic. Use fume hood."
        ))
        
        # Step 3: Filter solution
        steps.append(ProtocolStep(
            step_number=3,
            action="Filter solution",
            details="Filter through 0.45 µm PTFE syringe filter to remove particles.",
            duration="5 min",
            equipment=['filters'],
            critical=True
        ))
        
        # Step 4: Substrate preparation
        steps.append(ProtocolStep(
            step_number=4,
            action="Prepare substrates",
            details="Clean FTO glass with soap, water, acetone, isopropanol. UV-ozone treat for 15 min.",
            duration="30 min",
            equipment=['substrates', 'fume_hood']
        ))
        
        # Step 5: Spin-coating
        steps.append(ProtocolStep(
            step_number=5,
            action="Spin-coat perovskite",
            details="Load 40 µL solution on substrate. Spin at 1000 rpm (10 s) then 4000 rpm (30 s). Drop 100 µL chlorobenzene at t=25s (antisolvent dripping).",
            duration="1 min per substrate",
            equipment=['spin_coater'],
            safety_notes="Chlorobenzene is toxic. Use fume hood.",
            critical=True
        ))
        
        # Step 6: Annealing
        steps.append(ProtocolStep(
            step_number=6,
            action="Anneal films",
            details="Transfer to hotplate at 100°C. Anneal for 10 minutes. Cool to room temperature.",
            duration="15 min",
            equipment=['hotplate'],
            critical=True
        ))
        
        # Step 7: Quality check
        steps.append(ProtocolStep(
            step_number=7,
            action="Quality check",
            details="Inspect films visually. Should be mirror-like, uniform brown/black color. Measure thickness (~500 nm). XRD characterization recommended.",
            duration="10 min",
            equipment=[]
        ))
        
        return steps
    
    def _generate_two_step_protocol(self, parsed: Dict) -> List[ProtocolStep]:
        """Generate two-step sequential deposition protocol"""
        # Simplified - full implementation would have detailed two-step procedure
        return self._generate_one_step_protocol(parsed)
    
    def _generate_antisolvent_protocol(self, parsed: Dict) -> List[ProtocolStep]:
        """Generate antisolvent quenching protocol"""
        # Similar to one-step but with emphasis on inert atmosphere
        steps = self._generate_one_step_protocol(parsed)
        
        # Add inert atmosphere requirement
        for step in steps:
            if step.step_number >= 2:  # From dissolution onwards
                step.equipment.append('glovebox')
                step.safety_notes += " Perform under inert atmosphere (N2)."
        
        return steps
    
    def _generate_weighing_details(self, parsed: Dict) -> str:
        """Generate specific weighing instructions"""
        details = "In clean glass vials, weigh:\n"
        
        for element, ratio in parsed['a_site']:
            mass = ratio * 159.0  # Approximate molar mass
            precursor = self.precursors.get(element, {}).get('name', element)
            details += f"- {precursor}: {mass:.1f} mg (ratio {ratio:.2f})\n"
        
        for element, ratio in parsed['b_site']:
            mass = ratio * 461.0  # PbI2 molar mass
            precursor = self.precursors.get(element, {}).get('name', element)
            details += f"- {precursor}: {mass:.1f} mg\n"
        
        details += "\nTarget concentration: 1.3 M in 1 mL total volume"
        
        return details
    
    def _calculate_total_time(self, steps: List[ProtocolStep]) -> str:
        """Calculate total protocol time"""
        total_minutes = 0
        
        for step in steps:
            duration = step.duration
            if 'hour' in duration:
                hours = float(duration.split()[0])
                total_minutes += hours * 60
            elif 'min' in duration:
                minutes = float(duration.split()[0])
                total_minutes += minutes
        
        hours = int(total_minutes // 60)
        minutes = int(total_minutes % 60)
        
        if hours > 0:
            return f"{hours} hour{'s' if hours > 1 else ''} {minutes} min"
        else:
            return f"{minutes} min"
    
    def _calculate_total_cost(self, parsed: Dict) -> float:
        """Calculate total material cost"""
        cost = 0.0
        
        # Precursor costs (for 10 substrates)
        for element, ratio in parsed['a_site'] + parsed['b_site']:
            precursor_cost = self.precursors.get(element, {}).get('cost_per_g', 10.0)
            mass_needed = ratio * 0.5  # grams for 10 substrates
            cost += precursor_cost * mass_needed
        
        # Solvent costs
        cost += 10.0  # Solvents for 10 substrates
        
        # Substrate costs
        cost += 20.0  # FTO glass
        
        return round(cost, 2)
    
    def _compile_safety_warnings(self, parsed: Dict) -> List[str]:
        """Compile all safety warnings"""
        warnings = [
            "⚠️ GENERAL SAFETY:",
            "- Perform all work in chemical fume hood",
            "- Wear lab coat, gloves, and safety glasses at all times",
            "- No food or drinks in lab"
        ]
        
        # Lead hazards
        if any(elem == 'Pb' for elem, _ in parsed['b_site']):
            warnings.extend([
                "",
                "☠️ LEAD HAZARD:",
                "- Lead compounds are TOXIC and reproductive hazards",
                "- Use double gloves (nitrile recommended)",
                "- Wash hands thoroughly after handling",
                "- Dispose lead waste in designated hazardous waste container",
                "- Pregnant women should avoid handling lead compounds",
                "- Regular blood lead level monitoring recommended for frequent users"
            ])
        
        # Tin air sensitivity
        if any(elem == 'Sn' for elem, _ in parsed['b_site']):
            warnings.extend([
                "",
                "🔒 INERT ATMOSPHERE REQUIRED:",
                "- Tin(II) compounds oxidize rapidly in air",
                "- Handle in N2 or Ar glovebox (O2, H2O < 1 ppm)",
                "- Prepare fresh solutions daily",
                "- Store precursors in vacuum desiccator"
            ])
        
        # Solvent hazards
        warnings.extend([
            "",
            "🧪 SOLVENT HAZARDS:",
            "- DMF, DMSO: Reproductive hazards, skin absorption",
            "- Chlorobenzene: Toxic by inhalation and skin contact",
            "- Use fume hood, avoid skin contact"
        ])
        
        return warnings
    
    def _compile_equipment_list(self, steps: List[ProtocolStep]) -> List[str]:
        """Compile required equipment"""
        equipment_set = set()
        
        for step in steps:
            equipment_set.update(step.equipment)
        
        equipment_list = [self.equipment.get(eq, eq) for eq in sorted(equipment_set)]
        
        return equipment_list
    
    def _get_precursor_details(self, parsed: Dict) -> Dict[str, Dict]:
        """Get detailed precursor information"""
        details = {}
        
        for element, ratio in parsed['a_site'] + parsed['b_site']:
            if element in self.precursors:
                details[element] = self.precursors[element]
        
        return details
    
    def _generate_notes(self, parsed: Dict) -> str:
        """Generate additional notes"""
        notes = "**Important Notes:**\n\n"
        
        if parsed['mixed_halides']:
            notes += "- Mixed halide systems may show phase segregation under illumination\n"
            notes += "- Optimize annealing temperature (typically 100-120°C)\n"
        
        if parsed['lead_free']:
            notes += "- Sn-based perovskites are less stable than Pb analogues\n"
            notes += "- Minimize air exposure throughout entire process\n"
            notes += "- Consider adding SnF2 (2-10 mol%) to suppress Sn2+ → Sn4+ oxidation\n"
        
        notes += "\n**Troubleshooting:**\n\n"
        notes += "- Pinholes: Reduce spin speed or increase concentration\n"
        notes += "- Non-uniform film: Check substrate cleanliness, adjust antisolvent timing\n"
        notes += "- Poor crystallinity: Optimize annealing temperature/time\n"
        
        return notes
    
    def format_protocol(self, protocol: SynthesisProtocol, 
                       format: str = "markdown") -> str:
        """
        Format protocol for export
        
        Args:
            protocol: SynthesisProtocol object
            format: "markdown" or "pdf_ready"
        
        Returns:
            Formatted protocol string
        """
        if format == "markdown":
            return self._format_markdown(protocol)
        elif format == "pdf_ready":
            return self._format_pdf_ready(protocol)
    
    def _format_markdown(self, p: SynthesisProtocol) -> str:
        """Format protocol as Markdown"""
        md = f"""# {p.title}

**Composition:** {p.composition}  
**Estimated Time:** {p.total_time}  
**Estimated Cost:** ${p.total_cost} (materials for 10 substrates)  
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}

---

## ⚠️ Safety Warnings

{chr(10).join(p.safety_warnings)}

---

## 📋 Equipment Checklist

"""
        for i, eq in enumerate(p.equipment_list, 1):
            md += f"{i}. {eq}\n"
        
        md += "\n---\n\n## 🧪 Precursors\n\n"
        md += "| Component | Supplier | Purity | Cost/g |\n"
        md += "|-----------|----------|--------|--------|\n"
        
        for elem, details in p.precursors.items():
            md += f"| {details['name']} | {details['supplier']} | {details['purity']} | ${details['cost_per_g']} |\n"
        
        md += "\n---\n\n## 📝 Protocol Steps\n\n"
        
        for step in p.steps:
            critical = "⭐ CRITICAL" if step.critical else ""
            md += f"### Step {step.step_number}: {step.action} {critical}\n\n"
            md += f"**Duration:** {step.duration}\n\n"
            md += f"**Procedure:**\n{step.details}\n\n"
            
            if step.safety_notes:
                md += f"⚠️ **Safety:** {step.safety_notes}\n\n"
            
            md += "---\n\n"
        
        md += f"## 📌 Additional Notes\n\n{p.notes}\n"
        
        md += f"""
---

**Protocol generated by AlphaMaterials V10**  
*Always review safety data sheets (SDS) before handling chemicals*  
*Consult with experienced researchers before first synthesis*
"""
        
        return md
    
    def _format_pdf_ready(self, p: SynthesisProtocol) -> str:
        """Format protocol for PDF export (HTML-based)"""
        # Similar to markdown but with HTML styling
        return self._format_markdown(p)


def demonstrate_protocol_generation():
    """Demonstrate protocol generation"""
    generator = ProtocolGenerator()
    
    # Test compositions
    compositions = [
        "MAPbI3",  # Simple
        "Cs0.1FA0.9PbI2.8Br0.2",  # Complex mixed
        "MASnI3"  # Lead-free
    ]
    
    protocols = {}
    
    for comp in compositions:
        protocol = generator.generate_protocol(comp)
        formatted = generator.format_protocol(protocol, format="markdown")
        protocols[comp] = formatted
    
    return protocols
