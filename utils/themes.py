"""
AlphaMaterials V11: Theme Management + Accessibility
=====================================================
Light/Dark theme toggle, colorblind-safe palettes, font controls.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ThemeConfig:
    """Theme configuration."""
    name: str
    bg_primary: str
    bg_secondary: str
    bg_card: str
    text_primary: str
    text_secondary: str
    accent: str
    accent_secondary: str
    success: str
    warning: str
    error: str
    border: str
    chart_colors: List[str] = field(default_factory=list)


DARK_THEME = ThemeConfig(
    name="dark",
    bg_primary="#0a0e1a",
    bg_secondary="#1a1f2e",
    bg_card="#141824",
    text_primary="#e0e0e0",
    text_secondary="#a0a0a0",
    accent="#00d4aa",
    accent_secondary="#0088cc",
    success="#00c853",
    warning="#ffd600",
    error="#ff1744",
    border="#2a2f3e",
    chart_colors=["#00d4aa", "#0088cc", "#ff6b6b", "#ffd93d", "#6c5ce7",
                  "#a29bfe", "#fd79a8", "#00b894", "#e17055", "#74b9ff"]
)

LIGHT_THEME = ThemeConfig(
    name="light",
    bg_primary="#ffffff",
    bg_secondary="#f5f7fa",
    bg_card="#ffffff",
    text_primary="#1a1a2e",
    text_secondary="#666666",
    accent="#028090",
    accent_secondary="#00a896",
    success="#2e7d32",
    warning="#f57f17",
    error="#c62828",
    border="#e0e0e0",
    chart_colors=["#028090", "#00a896", "#e74c3c", "#f39c12", "#8e44ad",
                  "#3498db", "#e91e63", "#27ae60", "#d35400", "#2980b9"]
)

# Colorblind-safe palettes (Wong, 2011 - Nature Methods)
COLORBLIND_SAFE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
                   "#0072B2", "#D55E00", "#CC79A7", "#999999"]

HIGH_CONTRAST = ThemeConfig(
    name="high_contrast",
    bg_primary="#000000",
    bg_secondary="#1a1a1a",
    bg_card="#0d0d0d",
    text_primary="#ffffff",
    text_secondary="#cccccc",
    accent="#00ff88",
    accent_secondary="#00aaff",
    success="#00ff00",
    warning="#ffff00",
    error="#ff0000",
    border="#444444",
    chart_colors=COLORBLIND_SAFE
)

THEMES = {"dark": DARK_THEME, "light": LIGHT_THEME, "high_contrast": HIGH_CONTRAST}


def get_theme(name: str = "dark") -> ThemeConfig:
    return THEMES.get(name, DARK_THEME)


def get_chart_colors(colorblind_safe: bool = False) -> List[str]:
    if colorblind_safe:
        return COLORBLIND_SAFE
    return DARK_THEME.chart_colors


def generate_css(theme: ThemeConfig, font_size: int = 14) -> str:
    """Generate custom CSS for Streamlit."""
    return f"""
    <style>
    .stApp {{
        background-color: {theme.bg_primary};
        color: {theme.text_primary};
    }}
    .stSidebar {{
        background-color: {theme.bg_secondary};
    }}
    .stMetric {{
        background-color: {theme.bg_card};
        border: 1px solid {theme.border};
        border-radius: 8px;
        padding: 1rem;
    }}
    .stMarkdown p, .stMarkdown li {{
        font-size: {font_size}px;
    }}
    h1, h2, h3 {{
        color: {theme.accent};
    }}
    .success {{ color: {theme.success}; }}
    .warning {{ color: {theme.warning}; }}
    .error {{ color: {theme.error}; }}
    </style>
    """


def apply_theme(st_module, theme_name: str = "dark", font_size: int = 14):
    """Apply theme to Streamlit app."""
    theme = get_theme(theme_name)
    st_module.markdown(generate_css(theme, font_size), unsafe_allow_html=True)
    return theme
