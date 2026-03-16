#!/usr/bin/env python3
"""
Theme & Accessibility Manager
==============================

Manage app themes and accessibility features:
- Light/Dark theme toggle
- Colorblind-safe palettes
- Font size controls
- High-contrast mode
- Mobile-responsive hints

Part of AlphaMaterials V11
"""

from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass


class ThemeMode(Enum):
    """Theme modes"""
    DARK = "dark"
    LIGHT = "light"


class ColorblindMode(Enum):
    """Colorblind-safe palette modes"""
    NONE = "none"
    PROTANOPIA = "protanopia"  # Red-blind
    DEUTERANOPIA = "deuteranopia"  # Green-blind
    TRITANOPIA = "tritanopia"  # Blue-blind


class FontSize(Enum):
    """Font size presets"""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XLARGE = "xlarge"


@dataclass
class ThemeConfig:
    """Theme configuration"""
    mode: ThemeMode = ThemeMode.DARK
    colorblind_mode: ColorblindMode = ColorblindMode.NONE
    font_size: FontSize = FontSize.MEDIUM
    high_contrast: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "mode": self.mode.value,
            "colorblind_mode": self.colorblind_mode.value,
            "font_size": self.font_size.value,
            "high_contrast": self.high_contrast
        }


class ThemeManager:
    """
    Manage app themes and accessibility
    """
    
    # Color palettes
    PALETTES = {
        # Standard palettes
        "dark_standard": {
            "primary": "#3b82f6",  # Blue
            "secondary": "#8b5cf6",  # Purple
            "accent": "#ec4899",  # Pink
            "success": "#10b981",  # Green
            "warning": "#f59e0b",  # Amber
            "danger": "#ef4444",  # Red
            "background": "#0a0e1a",
            "surface": "#1e2130",
            "text": "#fafafa",
            "text_secondary": "#b0b0b0"
        },
        
        "light_standard": {
            "primary": "#2563eb",  # Blue
            "secondary": "#7c3aed",  # Purple
            "accent": "#db2777",  # Pink
            "success": "#059669",  # Green
            "warning": "#d97706",  # Amber
            "danger": "#dc2626",  # Red
            "background": "#ffffff",
            "surface": "#f3f4f6",
            "text": "#1f2937",
            "text_secondary": "#6b7280"
        },
        
        # Colorblind-safe palettes (Okabe-Ito palette)
        "colorblind_safe": {
            "blue": "#0072B2",
            "orange": "#E69F00",
            "green": "#009E73",
            "yellow": "#F0E442",
            "purple": "#CC79A7",
            "cyan": "#56B4E9",
            "red": "#D55E00",
            "black": "#000000"
        },
        
        # High contrast
        "high_contrast_dark": {
            "primary": "#ffffff",
            "secondary": "#ffff00",
            "accent": "#00ffff",
            "success": "#00ff00",
            "warning": "#ffff00",
            "danger": "#ff0000",
            "background": "#000000",
            "surface": "#1a1a1a",
            "text": "#ffffff",
            "text_secondary": "#ffffff"
        },
        
        "high_contrast_light": {
            "primary": "#000000",
            "secondary": "#0000ff",
            "accent": "#ff00ff",
            "success": "#008000",
            "warning": "#ff8800",
            "danger": "#ff0000",
            "background": "#ffffff",
            "surface": "#f0f0f0",
            "text": "#000000",
            "text_secondary": "#000000"
        }
    }
    
    # Font size mappings (rem units)
    FONT_SIZES = {
        FontSize.SMALL: {
            "base": "0.875rem",
            "heading1": "2.0rem",
            "heading2": "1.5rem",
            "heading3": "1.25rem"
        },
        FontSize.MEDIUM: {
            "base": "1.0rem",
            "heading1": "2.5rem",
            "heading2": "1.875rem",
            "heading3": "1.5rem"
        },
        FontSize.LARGE: {
            "base": "1.125rem",
            "heading1": "3.0rem",
            "heading2": "2.25rem",
            "heading3": "1.75rem"
        },
        FontSize.XLARGE: {
            "base": "1.25rem",
            "heading1": "3.5rem",
            "heading2": "2.625rem",
            "heading3": "2.0rem"
        }
    }
    
    def __init__(self, config: Optional[ThemeConfig] = None):
        self.config = config or ThemeConfig()
    
    def get_palette(self) -> Dict[str, str]:
        """Get current color palette"""
        if self.config.high_contrast:
            # High contrast mode
            palette_key = f"high_contrast_{self.config.mode.value}"
            return self.PALETTES[palette_key]
        
        elif self.config.colorblind_mode != ColorblindMode.NONE:
            # Colorblind-safe palette
            return self.PALETTES["colorblind_safe"]
        
        else:
            # Standard palette
            palette_key = f"{self.config.mode.value}_standard"
            return self.PALETTES[palette_key]
    
    def get_font_sizes(self) -> Dict[str, str]:
        """Get current font sizes"""
        return self.FONT_SIZES[self.config.font_size]
    
    def generate_css(self) -> str:
        """
        Generate custom CSS for current theme
        """
        palette = self.get_palette()
        fonts = self.get_font_sizes()
        
        css = f"""
<style>
    :root {{
        /* Colors */
        --primary: {palette.get('primary', palette.get('blue', '#3b82f6'))};
        --secondary: {palette.get('secondary', palette.get('purple', '#8b5cf6'))};
        --accent: {palette.get('accent', palette.get('orange', '#ec4899'))};
        --success: {palette.get('success', palette.get('green', '#10b981'))};
        --warning: {palette.get('warning', palette.get('yellow', '#f59e0b'))};
        --danger: {palette.get('danger', palette.get('red', '#ef4444'))};
        --background: {palette.get('background', '#0a0e1a')};
        --surface: {palette.get('surface', '#1e2130')};
        --text: {palette.get('text', '#fafafa')};
        --text-secondary: {palette.get('text_secondary', '#b0b0b0')};
        
        /* Fonts */
        --font-base: {fonts['base']};
        --font-h1: {fonts['heading1']};
        --font-h2: {fonts['heading2']};
        --font-h3: {fonts['heading3']};
    }}
    
    .stApp {{
        background: var(--background);
        color: var(--text);
        font-size: var(--font-base);
    }}
    
    .main-title {{
        font-size: var(--font-h1);
        font-weight: 900;
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 50%, var(--accent) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }}
    
    .subtitle {{
        font-size: var(--font-h3);
        color: var(--text-secondary);
        text-align: center;
        margin-bottom: 2rem;
    }}
    
    .metric-card {{
        background: var(--surface);
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid var(--primary);
        margin: 0.5rem 0;
    }}
    
    .success-box {{
        background: color-mix(in srgb, var(--success) 20%, var(--background));
        border-left: 5px solid var(--success);
        border-radius: 10px;
        padding: 1.2rem;
        margin: 1rem 0;
    }}
    
    .warning-box {{
        background: color-mix(in srgb, var(--warning) 20%, var(--background));
        border-left: 5px solid var(--warning);
        border-radius: 10px;
        padding: 1.2rem;
        margin: 1rem 0;
    }}
    
    .danger-box {{
        background: color-mix(in srgb, var(--danger) 20%, var(--background));
        border-left: 5px solid var(--danger);
        border-radius: 10px;
        padding: 1.2rem;
        margin: 1rem 0;
    }}
    
    .info-box {{
        background: color-mix(in srgb, var(--primary) 20%, var(--background));
        border-left: 5px solid var(--primary);
        border-radius: 10px;
        padding: 1.2rem;
        margin: 1rem 0;
    }}
    
    /* Buttons */
    .stButton > button {{
        background: var(--primary);
        color: var(--text);
        border-radius: 8px;
        font-size: var(--font-base);
        padding: 0.5rem 1rem;
    }}
    
    .stButton > button:hover {{
        background: var(--secondary);
    }}
    
    /* Tables */
    .dataframe {{
        font-size: var(--font-base);
        background: var(--surface);
        color: var(--text);
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: var(--surface);
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: var(--text);
        font-size: var(--font-base);
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: var(--primary);
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: var(--surface);
    }}
    
    /* High contrast adjustments */
    {self._get_high_contrast_css() if self.config.high_contrast else ''}
    
    /* Mobile responsive */
    @media (max-width: 768px) {{
        .main-title {{
            font-size: calc(var(--font-h1) * 0.6);
        }}
        
        .subtitle {{
            font-size: calc(var(--font-h3) * 0.8);
        }}
        
        .metric-card {{
            padding: 0.8rem;
        }}
    }}
</style>
"""
        return css
    
    def _get_high_contrast_css(self) -> str:
        """Additional CSS for high contrast mode"""
        return """
    /* High Contrast Mode */
    * {
        border-color: var(--text) !important;
    }
    
    .stButton > button {
        border: 2px solid var(--text);
        font-weight: bold;
    }
    
    a {
        text-decoration: underline;
        font-weight: bold;
    }
"""
    
    def get_plotly_colors(self) -> List[str]:
        """
        Get Plotly color sequence for current theme
        
        Returns colorblind-safe sequence if enabled
        """
        if self.config.colorblind_mode != ColorblindMode.NONE:
            # Okabe-Ito colorblind-safe palette
            return [
                "#0072B2",  # Blue
                "#E69F00",  # Orange
                "#009E73",  # Green
                "#F0E442",  # Yellow
                "#CC79A7",  # Purple
                "#56B4E9",  # Cyan
                "#D55E00",  # Red
                "#000000"   # Black
            ]
        else:
            # Standard vibrant colors
            palette = self.get_palette()
            return [
                palette.get('primary', '#3b82f6'),
                palette.get('secondary', '#8b5cf6'),
                palette.get('accent', '#ec4899'),
                palette.get('success', '#10b981'),
                palette.get('warning', '#f59e0b'),
                palette.get('danger', '#ef4444')
            ]
    
    def get_plotly_template(self) -> Dict:
        """
        Get Plotly template configuration for current theme
        """
        palette = self.get_palette()
        colors = self.get_plotly_colors()
        
        template = {
            "layout": {
                "colorway": colors,
                "paper_bgcolor": palette.get('background', '#0a0e1a'),
                "plot_bgcolor": palette.get('surface', '#1e2130'),
                "font": {
                    "color": palette.get('text', '#fafafa'),
                    "size": 12 if self.config.font_size == FontSize.SMALL else
                           14 if self.config.font_size == FontSize.MEDIUM else
                           16 if self.config.font_size == FontSize.LARGE else 18
                },
                "xaxis": {
                    "gridcolor": palette.get('text_secondary', '#b0b0b0'),
                    "linecolor": palette.get('text', '#fafafa'),
                    "zerolinecolor": palette.get('text_secondary', '#b0b0b0')
                },
                "yaxis": {
                    "gridcolor": palette.get('text_secondary', '#b0b0b0'),
                    "linecolor": palette.get('text', '#fafafa'),
                    "zerolinecolor": palette.get('text_secondary', '#b0b0b0')
                }
            }
        }
        
        return template
    
    def update_config(
        self,
        mode: Optional[ThemeMode] = None,
        colorblind_mode: Optional[ColorblindMode] = None,
        font_size: Optional[FontSize] = None,
        high_contrast: Optional[bool] = None
    ):
        """Update theme configuration"""
        if mode is not None:
            self.config.mode = mode
        if colorblind_mode is not None:
            self.config.colorblind_mode = colorblind_mode
        if font_size is not None:
            self.config.font_size = font_size
        if high_contrast is not None:
            self.config.high_contrast = high_contrast


def demonstrate_themes():
    """Demonstrate theme manager"""
    print("🎨 AlphaMaterials Theme Manager Demo\n")
    
    # Dark theme (default)
    print("🌙 Dark Theme (Standard)")
    manager = ThemeManager()
    palette = manager.get_palette()
    print(f"  Background: {palette['background']}")
    print(f"  Primary: {palette['primary']}")
    print(f"  Text: {palette['text']}")
    
    # Light theme
    print("\n☀️ Light Theme")
    manager.update_config(mode=ThemeMode.LIGHT)
    palette = manager.get_palette()
    print(f"  Background: {palette['background']}")
    print(f"  Primary: {palette['primary']}")
    print(f"  Text: {palette['text']}")
    
    # Colorblind-safe
    print("\n🌈 Colorblind-Safe Palette")
    manager.update_config(colorblind_mode=ColorblindMode.DEUTERANOPIA)
    colors = manager.get_plotly_colors()
    print(f"  Colors: {colors[:4]}")
    
    # High contrast
    print("\n⚡ High Contrast Mode (Dark)")
    manager.update_config(
        mode=ThemeMode.DARK,
        colorblind_mode=ColorblindMode.NONE,
        high_contrast=True
    )
    palette = manager.get_palette()
    print(f"  Background: {palette['background']}")
    print(f"  Primary: {palette['primary']}")
    print(f"  Text: {palette['text']}")
    
    # Font sizes
    print("\n📝 Font Size Options:")
    for size in FontSize:
        manager.update_config(font_size=size, high_contrast=False)
        fonts = manager.get_font_sizes()
        print(f"  {size.value}: Base={fonts['base']}, H1={fonts['heading1']}")
    
    # Generate CSS sample
    print("\n🎨 Generated CSS Sample:")
    manager.update_config(
        mode=ThemeMode.DARK,
        font_size=FontSize.MEDIUM,
        colorblind_mode=ColorblindMode.NONE,
        high_contrast=False
    )
    css = manager.generate_css()
    print(css[:400] + "...")
    
    # Plotly template
    print("\n📊 Plotly Template:")
    template = manager.get_plotly_template()
    print(f"  Colors: {template['layout']['colorway'][:3]}")
    print(f"  Font size: {template['layout']['font']['size']}")


if __name__ == "__main__":
    demonstrate_themes()
