"""Professional Dash-based dashboard for strategy analysis."""

from .app import create_app, launch_dashboard
from .loaders import save_results, load_results

__all__ = ["create_app", "launch_dashboard", "save_results", "load_results"]