"""CLI for rlwatch — diagnose completed runs and launch dashboard.

Usage:
    rlwatch diagnose --log-dir ./rlwatch_logs/
    rlwatch dashboard --log-dir ./rlwatch_logs/
    rlwatch runs --log-dir ./rlwatch_logs/
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="rlwatch")
def main():
    """rlwatch — GRPO/PPO training instability detection."""
    pass


@main.command()
@click.option("--log-dir", default="./rlwatch_logs", help="Directory containing rlwatch logs")
@click.option("--run-id", default=None, help="Specific run ID to diagnose (default: latest)")
@click.option("--format", "output_format", type=click.Choice(["rich", "json"]), default="rich")
def diagnose(log_dir: str, run_id: str | None, output_format: str):
    """Retrospective analysis of a completed training run."""
    db_path = Path(log_dir) / "metrics.db"
    if not db_path.exists():
        console.print(f"[red]Error:[/] No rlwatch database found at {db_path}")
        console.print("Make sure you have completed at least one monitored training run.")
        raise SystemExit(1)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Get run
    if run_id is None:
        row = conn.execute("SELECT * FROM runs ORDER BY started_at DESC LIMIT 1").fetchone()
        if row is None:
            console.print("[red]No runs found in database.[/]")
            raise SystemExit(1)
        run_id = row["run_id"]
    else:
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            console.print(f"[red]Run {run_id} not found.[/]")
            raise SystemExit(1)

    run = dict(row)

    # Load metrics
    metrics = conn.execute(
        "SELECT * FROM metrics WHERE run_id = ? ORDER BY step", (run_id,)
    ).fetchall()
    metrics = [dict(m) for m in metrics]

    # Load alerts
    alerts = conn.execute(
        "SELECT * FROM alerts WHERE run_id = ? ORDER BY step", (run_id,)
    ).fetchall()
    alerts = [dict(a) for a in alerts]

    conn.close()

    if output_format == "json":
        _output_json(run, metrics, alerts)
    else:
        _output_rich(run, metrics, alerts)


def _output_json(run: dict, metrics: list[dict], alerts: list[dict]):
    """Output diagnosis as JSON."""
    diagnosis = _build_diagnosis(run, metrics, alerts)
    click.echo(json.dumps(diagnosis, indent=2, default=str))


def _build_diagnosis(run: dict, metrics: list[dict], alerts: list[dict]) -> dict:
    """Build a diagnosis report."""
    import numpy as np

    report = {
        "run_id": run["run_id"],
        "framework": run.get("framework", "unknown"),
        "started_at": datetime.fromtimestamp(run["started_at"]).isoformat() if run.get("started_at") else None,
        "total_steps": len(metrics),
        "total_alerts": len(alerts),
        "critical_alerts": sum(1 for a in alerts if a["severity"] == "critical"),
        "warning_alerts": sum(1 for a in alerts if a["severity"] == "warning"),
        "detectors_triggered": list(set(a["detector"] for a in alerts)),
        "alerts": alerts,
    }

    # Metric summaries
    if metrics:
        for metric_name in ["entropy", "kl_divergence", "reward_mean", "reward_std", "advantage_std", "loss", "grad_norm"]:
            values = [m[metric_name] for m in metrics if m.get(metric_name) is not None]
            if values:
                arr = np.array(values)
                report[f"{metric_name}_summary"] = {
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "first": float(arr[0]),
                    "last": float(arr[-1]),
                    "trend": "decreasing" if arr[-1] < arr[0] else "increasing",
                }

    # Overall health assessment
    if not alerts:
        report["health"] = "healthy"
        report["assessment"] = "No instabilities detected during this run."
    elif any(a["severity"] == "critical" for a in alerts):
        report["health"] = "critical"
        detectors = set(a["detector"] for a in alerts if a["severity"] == "critical")
        report["assessment"] = f"Critical instabilities detected: {', '.join(detectors)}."
    else:
        report["health"] = "warning"
        detectors = set(a["detector"] for a in alerts)
        report["assessment"] = f"Warning-level issues detected: {', '.join(detectors)}."

    return report


def _output_rich(run: dict, metrics: list[dict], alerts: list[dict]):
    """Output diagnosis with rich formatting."""
    import numpy as np

    diagnosis = _build_diagnosis(run, metrics, alerts)

    # Header
    health_color = {"healthy": "green", "warning": "yellow", "critical": "red"}.get(
        diagnosis["health"], "white"
    )
    console.print(Panel(
        f"[bold]Run:[/] {diagnosis['run_id']}\n"
        f"[bold]Framework:[/] {diagnosis['framework']}\n"
        f"[bold]Started:[/] {diagnosis.get('started_at', 'unknown')}\n"
        f"[bold]Total Steps:[/] {diagnosis['total_steps']}\n"
        f"[bold]Health:[/] [{health_color}]{diagnosis['health'].upper()}[/]\n"
        f"\n{diagnosis['assessment']}",
        title="[bold]rlwatch Diagnosis Report[/]",
        border_style=health_color,
    ))

    # Metric summaries
    if metrics:
        table = Table(title="Metric Summaries", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")
        table.add_column("Mean", justify="right")
        table.add_column("First", justify="right")
        table.add_column("Last", justify="right")
        table.add_column("Trend", justify="center")

        for metric_name in ["entropy", "kl_divergence", "reward_mean", "reward_std", "advantage_std", "loss", "grad_norm"]:
            key = f"{metric_name}_summary"
            if key in diagnosis:
                s = diagnosis[key]
                trend_icon = "[red]↓[/]" if s["trend"] == "decreasing" else "[green]↑[/]"
                table.add_row(
                    metric_name,
                    f"{s['min']:.4f}",
                    f"{s['max']:.4f}",
                    f"{s['mean']:.4f}",
                    f"{s['first']:.4f}",
                    f"{s['last']:.4f}",
                    trend_icon,
                )

        console.print(table)

    # Alerts
    if alerts:
        console.print()
        table = Table(title=f"Alerts ({len(alerts)} total)", show_header=True)
        table.add_column("Step", justify="right", style="bold")
        table.add_column("Detector", style="cyan")
        table.add_column("Severity")
        table.add_column("Message", max_width=80)
        table.add_column("Recommendation", max_width=60)

        for alert in alerts:
            sev_style = "red" if alert["severity"] == "critical" else "yellow"
            table.add_row(
                str(alert["step"]),
                alert["detector"],
                f"[{sev_style}]{alert['severity']}[/]",
                alert["message"][:80],
                alert["recommendation"][:60],
            )

        console.print(table)
    else:
        console.print("\n[green]No alerts triggered during this run.[/]")


@main.command()
@click.option("--log-dir", default="./rlwatch_logs", help="Directory containing rlwatch logs")
def runs(log_dir: str):
    """List all monitored training runs."""
    db_path = Path(log_dir) / "metrics.db"
    if not db_path.exists():
        console.print(f"[red]No rlwatch database found at {db_path}[/]")
        raise SystemExit(1)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    rows = conn.execute("SELECT * FROM runs ORDER BY started_at DESC").fetchall()
    conn.close()

    if not rows:
        console.print("[yellow]No runs found.[/]")
        return

    table = Table(title="Training Runs", show_header=True)
    table.add_column("Run ID", style="cyan")
    table.add_column("Framework")
    table.add_column("Started At")

    for row in rows:
        started = datetime.fromtimestamp(row["started_at"]).strftime("%Y-%m-%d %H:%M:%S") if row["started_at"] else "unknown"
        table.add_row(row["run_id"], row["framework"] or "unknown", started)

    console.print(table)


@main.command()
@click.option("--log-dir", default="./rlwatch_logs", help="Directory containing rlwatch logs")
@click.option("--port", default=8501, help="Dashboard port")
@click.option("--host", default="0.0.0.0", help="Dashboard host")
def dashboard(log_dir: str, port: int, host: str):
    """Launch the Streamlit monitoring dashboard."""
    dashboard_path = Path(__file__).parent / "dashboard.py"

    console.print(f"[bold green]Starting rlwatch dashboard...[/]")
    console.print(f"  Log dir: {log_dir}")
    console.print(f"  URL: http://{host}:{port}")

    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_path),
            "--server.port", str(port),
            "--server.address", host,
            "--server.headless", "true",
            "--",
            "--log-dir", log_dir,
        ],
        check=True,
    )


@main.command()
def init():
    """Generate a default rlwatch.yaml config file."""
    from rlwatch.config import RLWatchConfig

    import yaml

    config = RLWatchConfig()
    config_dict = config.to_dict()

    output_path = Path("rlwatch.yaml")
    if output_path.exists():
        console.print("[yellow]rlwatch.yaml already exists. Overwrite? [y/N][/]", end=" ")
        if input().strip().lower() != "y":
            console.print("Aborted.")
            return

    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    console.print(f"[green]Created {output_path}[/]")
    console.print("Edit this file to configure detection thresholds and alerts.")


if __name__ == "__main__":
    main()
