"""HTML report generation for the optimization workflow."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


def generate_html_report(
    output_path: str | Path,
    title: str,
    data_mode: str,
    dataset_name: str,
    row_count: int,
    model_metrics: dict[str, Any],
    original_params: dict[str, Any],
    optimized_params: dict[str, Any],
    original_strength: float,
    optimized_strength: float,
    improvement: float,
    ga_settings: dict[str, Any],
    honesty_notice: str,
    is_demo: bool,
) -> str:
    """Write a self-contained HTML report with the key project information."""
    report_title = "ILLUSTRATIVE DEMO" if is_demo else "Optimization Report"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_text = f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <title>{report_title}</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 24px; color: #1B2733; background: #F4F6F8; }}
          h1, h2 {{ color: #1F4E79; }}
          .card {{ background: white; border: 1px solid #D9DFE5; border-radius: 10px; padding: 16px; margin: 12px 0; }}
          .demo {{ background: #fff4e4; border-left: 4px solid #E8710A; padding: 12px; border-radius: 8px; }}
          table {{ border-collapse: collapse; width: 100%; }}
          th, td {{ border: 1px solid #D9DFE5; padding: 8px; text-align: left; }}
          th {{ background: #eef3f8; }}
        </style>
      </head>
      <body>
        <h1>{title}</h1>
        <div class="demo">{report_title}</div>
        <div class="card">
          <p><strong>Data mode:</strong> {data_mode}</p>
          <p><strong>Dataset:</strong> {dataset_name}</p>
          <p><strong>Rows:</strong> {row_count}</p>
          <p><strong>Generated:</strong> {timestamp}</p>
        </div>

        <div class="card">
          <h2>Model metrics</h2>
          <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Test R2</td><td>{model_metrics.get('test_r2', 'N/A')}</td></tr>
            <tr><td>CV R2</td><td>{model_metrics.get('cv_r2', 'N/A')}</td></tr>
            <tr><td>Train R2</td><td>{model_metrics.get('train_r2', 'N/A')}</td></tr>
            <tr><td>MAE</td><td>{model_metrics.get('mae', 'N/A')}</td></tr>
            <tr><td>RMSE</td><td>{model_metrics.get('rmse', 'N/A')}</td></tr>
            <tr><td>OOB R2</td><td>{model_metrics.get('oob_r2', 'N/A')}</td></tr>
          </table>
        </div>

        <div class="card">
          <h2>Original vs optimized parameters</h2>
          <table>
            <tr><th>Parameter</th><th>Original</th><th>Optimized</th></tr>
            {''.join(f'<tr><td>{k}</td><td>{v}</td><td>{optimized_params.get(k, "-")}</td></tr>' for k, v in original_params.items())}
          </table>
        </div>

        <div class="card">
          <h2>Strength comparison</h2>
          <p>Original MPa: {original_strength}</p>
          <p>Optimized MPa: {optimized_strength}</p>
          <p>Absolute improvement: {improvement}</p>
        </div>

        <div class="card">
          <h2>GA settings</h2>
          <table>
            <tr><th>Setting</th><th>Value</th></tr>
            {''.join(f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in ga_settings.items())}
          </table>
        </div>

        <div class="card">
          <h2>Honesty notice</h2>
          <p>{honesty_notice}</p>
        </div>
      </body>
    </html>
    """

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write("".join(html_text.split()), encoding="utf-8")
    return str(output)
