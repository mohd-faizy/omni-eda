"""Theme presets applied to matplotlib/seaborn plots and the HTML report's CSS."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class Theme:
    name: str
    palette: list[str]
    background: str
    foreground: str
    grid_color: str
    accent: str
    font_family: str = "'Segoe UI', Roboto, Helvetica, Arial, sans-serif"


THEMES: dict[str, Theme] = {
    "light": Theme(
        name="light",
        palette=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860", "#DA8BC3", "#8C8C8C"],
        background="#FFFFFF",
        foreground="#222222",
        grid_color="#E5E5E5",
        accent="#4C72B0",
    ),
    "dark": Theme(
        name="dark",
        palette=["#8AB4F8", "#F6AE7E", "#7FCB9E", "#F28B8C", "#B7A8E0", "#C9A98D", "#F0A6D6", "#B0B0B0"],
        background="#1E1E24",
        foreground="#EAEAEA",
        grid_color="#3A3A42",
        accent="#8AB4F8",
    ),
    "minimal": Theme(
        name="minimal",
        palette=["#333333", "#777777", "#AAAAAA", "#555555", "#999999", "#666666", "#888888", "#444444"],
        background="#FFFFFF",
        foreground="#111111",
        grid_color="#F0F0F0",
        accent="#111111",
    ),
    "corporate": Theme(
        name="corporate",
        palette=["#0B5FA5", "#00A99D", "#F2A93B", "#D64550", "#5D5D81", "#8FB339", "#6A4C93", "#4D4D4D"],
        background="#FAFBFC",
        foreground="#1A1A2E",
        grid_color="#E3E7EC",
        accent="#0B5FA5",
    ),
}


def get_theme(name: str) -> Theme:
    return THEMES.get(name, THEMES["light"])


def apply_theme(name: str = "light") -> Theme:
    """Apply a theme's colors/fonts to matplotlib's rcParams and return it."""
    theme = get_theme(name)
    plt.rcParams.update(
        {
            "figure.facecolor": theme.background,
            "axes.facecolor": theme.background,
            "axes.edgecolor": theme.foreground,
            "axes.labelcolor": theme.foreground,
            "axes.grid": True,
            "grid.color": theme.grid_color,
            "text.color": theme.foreground,
            "xtick.color": theme.foreground,
            "ytick.color": theme.foreground,
            "font.family": "sans-serif",
            "axes.prop_cycle": plt.cycler(color=theme.palette),
            "savefig.facecolor": theme.background,
        }
    )
    return theme
