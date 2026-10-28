"""Reusable Streamlit UI helpers for a clean engineering dashboard."""

from __future__ import annotations

import html
from typing import Any

import streamlit as st


def inject_theme_css() -> None:
    """Inject the custom CSS used across the app."""
    st.markdown(
        """
        <style>
        :root {
            --steel-blue: #1F4E79;
            --orange: #E8710A;
            --dark-slate: #14202E;
            --bg: #F4F6F8;
            --card-bg: #FFFFFF;
            --border: #D9DFE5;
            --text: #1B2733;
            --success: #2E7D32;
            --warning: #B26A00;
            --muted: #6B7C8F;
        }

        .stApp {
            background: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }

        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
        }

        .metric-card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 0.75rem;
            box-shadow: 0 1px 2px rgba(17, 24, 39, 0.04);
            padding: 0.9rem 1rem;
            min-height: 110px;
            margin-bottom: 0.75rem;
        }

        .metric-label {
            font-size: 0.74rem;
            color: var(--muted);
            font-weight: 600;
            letter-spacing: 0.02em;
            text-transform: uppercase;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            line-height: 1.2;
            margin-top: 0.25rem;
            color: var(--steel-blue);
        }

        .metric-unit {
            font-size: 0.75rem;
            margin-left: 0.2rem;
            color: var(--muted);
        }

        .metric-delta {
            margin-top: 0.2rem;
            font-size: 0.75rem;
            color: var(--success);
        }

        .section-header {
            border-bottom: 2px solid var(--steel-blue);
            padding-bottom: 0.2rem;
            margin-top: 1rem;
            margin-bottom: 0.75rem;
            font-weight: 700;
            color: var(--text);
        }

        .warning-box {
            border-left: 4px solid var(--warning);
            background: rgba(178, 106, 0, 0.06);
            padding: 0.75rem 0.9rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }

        .success-box {
            border-left: 4px solid var(--success);
            background: rgba(46, 125, 50, 0.06);
            padding: 0.75rem 0.9rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }

        .data-badge {
            display: inline-block;
            background: rgba(46, 125, 50, 0.12);
            color: var(--success);
            border: 1px solid rgba(46, 125, 50, 0.3);
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 600;
            padding: 0.28rem 0.7rem;
            margin-bottom: 0.5rem;
        }

        .demo-banner {
            background: rgba(232, 113, 10, 0.12);
            border: 1px solid rgba(232, 113, 10, 0.35);
            border-radius: 0.6rem;
            color: #5e3300;
            padding: 0.7rem 0.9rem;
            margin: 0.5rem 0 1rem 0;
            font-weight: 600;
        }

        table {
            font-size: 0.9rem;
        }

        .stDataFrame {
            border-radius: 0.5rem;
            overflow: hidden;
        }

        .sidebar .block-container {
            padding-top: 0.5rem;
        }

        .sidebar .stRadio, .sidebar .stSelectbox, .sidebar .stSlider, .sidebar .stNumberInput {
            margin-bottom: 0.6rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: Any, unit: str = "", delta: str | None = None) -> None:
    """Render a card-style metric with a label, value, optional unit and delta."""
    value_html = html.escape(str(value))
    unit_html = f'<span class="metric-unit">{html.escape(str(unit))}</span>' if unit else ""
    delta_html = f'<div class="metric-delta">{html.escape(str(delta))}</div>' if delta else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{html.escape(label)}</div>
            <div class="metric-value">{value_html}{unit_html}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_demo_banner() -> None:
    """Show a persistent warning banner in demo mode."""
    st.markdown(
        '<div class="demo-banner">ILLUSTRATIVE DEMO - synthetic data, NOT experimental measurements. Results are not validated.</div>',
        unsafe_allow_html=True,
    )


def render_data_badge(filename: str, rows: int) -> None:
    """Render the real-data badge used on every page in real mode."""
    st.markdown(f'<div class="data-badge">Real data: {filename}, {rows} rows</div>', unsafe_allow_html=True)


def render_footer() -> None:
    """Show a minimal footer matching the academic prototype style."""
    st.markdown("---")
    st.caption("Sri Eshwar College of Engineering | Prototype for academic review")
