"""Streamlit dashboard for rlwatch metric visualization.

Run with: streamlit run src/rlwatch/dashboard.py -- --log-dir ./rlwatch_logs
Or via CLI: rlwatch dashboard --log-dir ./rlwatch_logs
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


def get_log_dir() -> str:
    """Get log directory from CLI args or default."""
    # Check streamlit args (after --)
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--log-dir" and i + 1 < len(args):
            return args[i + 1]
    return "./rlwatch_logs"


def open_db(log_dir: str) -> sqlite3.Connection:
    """Open the rlwatch SQLite database."""
    db_path = Path(log_dir) / "metrics.db"
    if not db_path.exists():
        st.error(f"No rlwatch database found at {db_path}")
        st.info("Start a monitored training run first, or specify --log-dir")
        st.stop()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def load_runs(conn: sqlite3.Connection) -> list[dict]:
    """Load all runs from the database."""
    cursor = conn.execute("SELECT * FROM runs ORDER BY started_at DESC")
    return [dict(row) for row in cursor.fetchall()]


def load_metrics(conn: sqlite3.Connection, run_id: str) -> pd.DataFrame:
    """Load metrics for a specific run."""
    df = pd.read_sql_query(
        "SELECT * FROM metrics WHERE run_id = ? ORDER BY step",
        conn,
        params=(run_id,),
    )
    return df


def load_alerts(conn: sqlite3.Connection, run_id: str) -> pd.DataFrame:
    """Load alerts for a specific run."""
    df = pd.read_sql_query(
        "SELECT * FROM alerts WHERE run_id = ? ORDER BY step",
        conn,
        params=(run_id,),
    )
    return df


def load_config_for_run(conn: sqlite3.Connection, run_id: str) -> dict:
    """Load config for a specific run."""
    cursor = conn.execute("SELECT config_json FROM runs WHERE run_id = ?", (run_id,))
    row = cursor.fetchone()
    if row and row["config_json"]:
        return json.loads(row["config_json"])
    return {}


def create_metric_chart(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    threshold: float | None = None,
    threshold_label: str = "Threshold",
    alerts_df: pd.DataFrame | None = None,
    detector_name: str = "",
) -> go.Figure:
    """Create a plotly chart for a single metric."""
    fig = go.Figure()

    # Main metric line
    valid = df[df[metric_col].notna()]
    if valid.empty:
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title=title, height=300)
        return fig

    fig.add_trace(go.Scatter(
        x=valid["step"],
        y=valid[metric_col],
        mode="lines",
        name=metric_col,
        line=dict(color="#1f77b4", width=2),
    ))

    # Threshold line
    if threshold is not None:
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=threshold_label,
            annotation_position="top right",
        )

    # Alert markers
    if alerts_df is not None and not alerts_df.empty and detector_name:
        detector_alerts = alerts_df[alerts_df["detector"] == detector_name]
        if not detector_alerts.empty:
            # Get metric values at alert steps
            alert_steps = detector_alerts["step"].tolist()
            alert_values = []
            for s in alert_steps:
                matching = valid[valid["step"] == s]
                if not matching.empty:
                    alert_values.append(matching[metric_col].iloc[0])
                else:
                    alert_values.append(None)

            # Filter out None values
            valid_alerts = [(s, v) for s, v in zip(alert_steps, alert_values) if v is not None]
            if valid_alerts:
                steps, values = zip(*valid_alerts)
                severities = detector_alerts[detector_alerts["step"].isin(steps)]["severity"].tolist()
                colors = ["red" if s == "critical" else "orange" for s in severities]
                fig.add_trace(go.Scatter(
                    x=list(steps),
                    y=list(values),
                    mode="markers",
                    name="Alerts",
                    marker=dict(color=colors, size=10, symbol="x"),
                ))

    fig.update_layout(
        title=title,
        xaxis_title="Step",
        yaxis_title=metric_col,
        height=300,
        margin=dict(l=50, r=20, t=40, b=40),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def main():
    st.set_page_config(
        page_title="rlwatch Dashboard",
        page_icon=":chart_with_upwards_trend:",
        layout="wide",
    )

    st.title("rlwatch Dashboard")
    st.caption("Real-time GRPO/PPO training instability detection")

    log_dir = get_log_dir()
    conn = open_db(log_dir)

    # Sidebar: run selector
    runs = load_runs(conn)
    if not runs:
        st.warning("No training runs found in the database.")
        st.info(f"Looking in: {Path(log_dir).resolve()}")
        st.stop()

    st.sidebar.header("Select Run")
    run_options = {
        f"{r['run_id']} ({datetime.fromtimestamp(r['started_at']).strftime('%Y-%m-%d %H:%M')})"
        if r.get('started_at') else r['run_id']: r['run_id']
        for r in runs
    }
    selected_label = st.sidebar.selectbox("Training Run", options=list(run_options.keys()))
    selected_run_id = run_options[selected_label]

    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=False)
    if auto_refresh:
        st.rerun()

    # Load data for selected run
    metrics_df = load_metrics(conn, selected_run_id)
    alerts_df = load_alerts(conn, selected_run_id)
    config = load_config_for_run(conn, selected_run_id)

    # Run summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Steps", len(metrics_df))
    with col2:
        st.metric("Total Alerts", len(alerts_df))
    with col3:
        critical_count = len(alerts_df[alerts_df["severity"] == "critical"]) if not alerts_df.empty else 0
        st.metric("Critical Alerts", critical_count)
    with col4:
        warning_count = len(alerts_df[alerts_df["severity"] == "warning"]) if not alerts_df.empty else 0
        st.metric("Warnings", warning_count)

    if metrics_df.empty:
        st.info("No metrics recorded yet for this run. Waiting for training to begin...")
        st.stop()

    # Extract thresholds from config
    entropy_threshold = config.get("entropy_collapse", {}).get("threshold", 1.0)
    kl_sigma = config.get("kl_explosion", {}).get("sigma_multiplier", 3.0)
    reward_var_mult = config.get("reward_hacking", {}).get("variance_multiplier", 3.0)
    adv_std_mult = config.get("advantage_variance", {}).get("std_multiplier", 3.0)

    # Metric charts
    st.header("Training Metrics")

    # Row 1: Entropy and KL Divergence
    col1, col2 = st.columns(2)
    with col1:
        fig = create_metric_chart(
            metrics_df, "entropy", "Policy Entropy",
            threshold=entropy_threshold,
            threshold_label=f"Collapse threshold ({entropy_threshold})",
            alerts_df=alerts_df,
            detector_name="entropy_collapse",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = create_metric_chart(
            metrics_df, "kl_divergence", "KL Divergence",
            alerts_df=alerts_df,
            detector_name="kl_explosion",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Row 2: Reward and Advantage
    col1, col2 = st.columns(2)
    with col1:
        # Reward chart with mean and std
        fig = go.Figure()
        valid = metrics_df[metrics_df["reward_mean"].notna()]
        if not valid.empty:
            fig.add_trace(go.Scatter(
                x=valid["step"], y=valid["reward_mean"],
                mode="lines", name="Reward Mean",
                line=dict(color="#2ca02c", width=2),
            ))
            if "reward_std" in valid.columns and valid["reward_std"].notna().any():
                upper = valid["reward_mean"] + valid["reward_std"]
                lower = valid["reward_mean"] - valid["reward_std"]
                fig.add_trace(go.Scatter(
                    x=pd.concat([valid["step"], valid["step"][::-1]]),
                    y=pd.concat([upper, lower[::-1]]),
                    fill="toself",
                    fillcolor="rgba(44, 160, 44, 0.1)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="Reward +/- 1 Std",
                ))
        fig.update_layout(
            title="Reward Distribution",
            xaxis_title="Step", yaxis_title="Reward",
            height=300, margin=dict(l=50, r=20, t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = create_metric_chart(
            metrics_df, "advantage_std", "Advantage Std Dev",
            alerts_df=alerts_df,
            detector_name="advantage_variance",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Row 3: Loss and Learning Rate
    col1, col2 = st.columns(2)
    with col1:
        fig = create_metric_chart(metrics_df, "loss", "Training Loss")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = create_metric_chart(metrics_df, "learning_rate", "Learning Rate")
        st.plotly_chart(fig, use_container_width=True)

    # Alerts table
    st.header("Alert History")
    if alerts_df.empty:
        st.success("No alerts triggered for this run.")
    else:
        # Format alerts for display
        display_df = alerts_df[["step", "detector", "severity", "message", "recommendation"]].copy()
        display_df.columns = ["Step", "Detector", "Severity", "Message", "Recommendation"]

        # Color-code severity
        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "Severity": st.column_config.TextColumn(
                    "Severity",
                    help="Alert severity level",
                ),
            },
        )

    # Run config (expandable)
    with st.expander("Run Configuration"):
        st.json(config)

    conn.close()


if __name__ == "__main__":
    main()
