"""Streamlit dashboard for rlwatch metric visualization.

Run with: streamlit run src/rlwatch/dashboard.py -- --log-dir ./rlwatch_logs
Or via CLI: rlwatch dashboard --log-dir ./rlwatch_logs

Features (v0.5.0):
- Single-run view with 6 metric charts + alert overlay markers
- Run comparison view: overlay 2+ runs on the same charts
- Alert timeline: scatter chart of alerts by step × detector
- CSV/Parquet export for metrics and alerts
- Auto-refresh toggle (5-second interval)
"""

from __future__ import annotations

import io
import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Qualitative color palette for comparison mode (up to 10 runs).
_COMPARISON_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def get_log_dir() -> str:
    """Get log directory from CLI args or default."""
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
    cursor = conn.execute("SELECT * FROM runs ORDER BY started_at DESC")
    return [dict(row) for row in cursor.fetchall()]


def load_metrics(conn: sqlite3.Connection, run_id: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM metrics WHERE run_id = ? ORDER BY step",
        conn, params=(run_id,),
    )


def load_alerts(conn: sqlite3.Connection, run_id: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM alerts WHERE run_id = ? ORDER BY step",
        conn, params=(run_id,),
    )


def load_config_for_run(conn: sqlite3.Connection, run_id: str) -> dict:
    cursor = conn.execute("SELECT config_json FROM runs WHERE run_id = ?", (run_id,))
    row = cursor.fetchone()
    if row and row["config_json"]:
        return json.loads(row["config_json"])
    return {}


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------
def create_metric_chart(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    threshold: float | None = None,
    threshold_label: str = "Threshold",
    alerts_df: pd.DataFrame | None = None,
    detector_name: str = "",
    comparison_traces: list[tuple[str, pd.DataFrame, str]] | None = None,
) -> go.Figure:
    """Create a Plotly chart for a single metric.

    In comparison mode, ``comparison_traces`` is a list of
    ``(run_id, metrics_df, color)`` tuples. Each run gets its own trace
    with a distinct color. Alert markers are omitted in comparison mode
    to avoid clutter.
    """
    fig = go.Figure()

    if comparison_traces:
        # --- Comparison mode ---
        for run_id, run_df, color in comparison_traces:
            valid = run_df[run_df[metric_col].notna()] if metric_col in run_df.columns else pd.DataFrame()
            if valid.empty:
                continue
            fig.add_trace(go.Scatter(
                x=valid["step"],
                y=valid[metric_col],
                mode="lines",
                name=run_id,
                line=dict(color=color, width=2),
            ))
    else:
        # --- Single-run mode ---
        valid = df[df[metric_col].notna()] if metric_col in df.columns else pd.DataFrame()
        if valid.empty:
            fig.add_annotation(
                text="No data available",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            )
            fig.update_layout(title=title, height=300)
            return fig

        fig.add_trace(go.Scatter(
            x=valid["step"],
            y=valid[metric_col],
            mode="lines",
            name=metric_col,
            line=dict(color="#1f77b4", width=2),
        ))

        # Alert markers (single-run only)
        if alerts_df is not None and not alerts_df.empty and detector_name:
            detector_alerts = alerts_df[alerts_df["detector"] == detector_name]
            if not detector_alerts.empty:
                alert_steps = detector_alerts["step"].tolist()
                alert_values = []
                for s in alert_steps:
                    matching = valid[valid["step"] == s]
                    alert_values.append(
                        matching[metric_col].iloc[0] if not matching.empty else None
                    )
                valid_alerts = [
                    (s, v) for s, v in zip(alert_steps, alert_values) if v is not None
                ]
                if valid_alerts:
                    steps, values = zip(*valid_alerts)
                    severities = detector_alerts[
                        detector_alerts["step"].isin(steps)
                    ]["severity"].tolist()
                    colors = [
                        "red" if sev == "critical" else "orange" for sev in severities
                    ]
                    fig.add_trace(go.Scatter(
                        x=list(steps),
                        y=list(values),
                        mode="markers",
                        name="Alerts",
                        marker=dict(color=colors, size=10, symbol="x"),
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


def create_alert_timeline(
    alerts_df: pd.DataFrame,
    run_id: str = "",
    comparison_data: list[tuple[str, pd.DataFrame, str]] | None = None,
) -> go.Figure:
    """Create a scatter chart: X = step, Y = detector (categorical).

    Red dots for critical, orange for warning. Hover shows the alert message.
    In comparison mode, dots are colored per run instead of per severity.
    """
    fig = go.Figure()

    if comparison_data:
        for rid, adf, color in comparison_data:
            if adf.empty:
                continue
            fig.add_trace(go.Scatter(
                x=adf["step"],
                y=[f"{rid} — {d}" for d in adf["detector"]],
                mode="markers",
                name=rid,
                marker=dict(color=color, size=12, symbol="circle"),
                hovertext=[
                    m[:100] for m in adf["message"]
                ],
                hoverinfo="text+x",
            ))
    else:
        if alerts_df.empty:
            return fig
        colors = [
            "red" if sev == "critical" else "orange"
            for sev in alerts_df["severity"]
        ]
        fig.add_trace(go.Scatter(
            x=alerts_df["step"],
            y=alerts_df["detector"],
            mode="markers",
            name="Alerts",
            marker=dict(color=colors, size=12, symbol="circle"),
            hovertext=[
                m[:100] for m in alerts_df["message"]
            ],
            hoverinfo="text+x",
        ))

    fig.update_layout(
        title="Alert Timeline",
        xaxis_title="Step",
        yaxis_title="Detector",
        height=max(200, 60 * len(fig.data)),
        margin=dict(l=150, r=20, t=40, b=40),
        showlegend=bool(comparison_data),
    )
    return fig


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
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

    # --- Sidebar ---
    runs = load_runs(conn)
    if not runs:
        st.warning("No training runs found in the database.")
        st.info(f"Looking in: {Path(log_dir).resolve()}")
        st.stop()

    st.sidebar.header("Select Run")
    run_labels = {
        (
            f"{r['run_id']} ({datetime.fromtimestamp(r['started_at']).strftime('%Y-%m-%d %H:%M')})"
            if r.get("started_at")
            else r["run_id"]
        ): r["run_id"]
        for r in runs
    }
    selected_label = st.sidebar.selectbox(
        "Training Run", options=list(run_labels.keys())
    )
    selected_run_id = run_labels[selected_label]

    # Comparison mode: multiselect for overlaying runs.
    st.sidebar.markdown("---")
    compare_labels = st.sidebar.multiselect(
        "Compare Runs (overlay)",
        options=list(run_labels.keys()),
        help="Select 2+ runs to overlay their metrics on the same charts.",
    )
    compare_run_ids = [run_labels[lbl] for lbl in compare_labels]
    comparison_mode = len(compare_run_ids) >= 2

    # Auto-refresh toggle (fixed: sleep before rerun so it's not an infinite loop).
    auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=False)

    # --- Load data ---
    metrics_df = load_metrics(conn, selected_run_id)
    alerts_df = load_alerts(conn, selected_run_id)
    config = load_config_for_run(conn, selected_run_id)

    # Load comparison data if in comparison mode.
    comparison_metrics: list[tuple[str, pd.DataFrame, str]] = []
    comparison_alerts: list[tuple[str, pd.DataFrame, str]] = []
    if comparison_mode:
        for i, rid in enumerate(compare_run_ids):
            color = _COMPARISON_COLORS[i % len(_COMPARISON_COLORS)]
            comparison_metrics.append((rid, load_metrics(conn, rid), color))
            comparison_alerts.append((rid, load_alerts(conn, rid), color))

    # --- Summary metrics ---
    if comparison_mode:
        cols = st.columns(len(compare_run_ids))
        for i, (rid, mdf, _) in enumerate(comparison_metrics):
            adf = comparison_alerts[i][1]
            with cols[i]:
                st.markdown(f"**{rid}**")
                st.metric("Steps", len(mdf))
                st.metric("Alerts", len(adf))
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Steps", len(metrics_df))
        with col2:
            st.metric("Total Alerts", len(alerts_df))
        with col3:
            critical_count = (
                len(alerts_df[alerts_df["severity"] == "critical"])
                if not alerts_df.empty else 0
            )
            st.metric("Critical Alerts", critical_count)
        with col4:
            warning_count = (
                len(alerts_df[alerts_df["severity"] == "warning"])
                if not alerts_df.empty else 0
            )
            st.metric("Warnings", warning_count)

    if not comparison_mode and metrics_df.empty:
        st.info("No metrics recorded yet for this run. Waiting for training to begin...")
        st.stop()

    # --- Thresholds from config ---
    entropy_threshold = config.get("entropy_collapse", {}).get("threshold", 1.0)

    # --- Metric charts ---
    st.header("Training Metrics")

    chart_specs = [
        ("entropy", "Policy Entropy", entropy_threshold, f"Collapse threshold ({entropy_threshold})", "entropy_collapse"),
        ("kl_divergence", "KL Divergence", None, "", "kl_explosion"),
        ("reward_mean", "Reward Mean", None, "", ""),
        ("advantage_std", "Advantage Std Dev", None, "", "advantage_variance"),
        ("loss", "Training Loss", None, "", ""),
        ("learning_rate", "Learning Rate", None, "", ""),
        ("grad_norm", "Gradient Norm", None, "", "gradient_norm_spike"),
    ]

    # Render charts in rows of 2.
    for row_start in range(0, len(chart_specs), 2):
        cols = st.columns(2)
        for col_idx, spec in enumerate(chart_specs[row_start : row_start + 2]):
            metric_col, title, threshold, threshold_label, detector = spec
            with cols[col_idx]:
                fig = create_metric_chart(
                    metrics_df,
                    metric_col,
                    title,
                    threshold=threshold,
                    threshold_label=threshold_label,
                    alerts_df=alerts_df,
                    detector_name=detector,
                    comparison_traces=comparison_metrics if comparison_mode else None,
                )
                st.plotly_chart(fig, use_container_width=True)

    # --- Export buttons ---
    st.markdown("---")
    export_cols = st.columns(3)
    if comparison_mode:
        all_metrics = pd.concat(
            [df.assign(run_id=rid) for rid, df, _ in comparison_metrics],
            ignore_index=True,
        )
        all_alerts = pd.concat(
            [df.assign(run_id=rid) for rid, df, _ in comparison_alerts],
            ignore_index=True,
        )
    else:
        all_metrics = metrics_df
        all_alerts = alerts_df

    with export_cols[0]:
        st.download_button(
            "Download metrics (CSV)",
            data=all_metrics.to_csv(index=False),
            file_name="rlwatch_metrics.csv",
            mime="text/csv",
        )
    with export_cols[1]:
        parquet_buf = io.BytesIO()
        all_metrics.to_parquet(parquet_buf, index=False)
        st.download_button(
            "Download metrics (Parquet)",
            data=parquet_buf.getvalue(),
            file_name="rlwatch_metrics.parquet",
            mime="application/octet-stream",
        )
    with export_cols[2]:
        if not all_alerts.empty:
            st.download_button(
                "Download alerts (CSV)",
                data=all_alerts.to_csv(index=False),
                file_name="rlwatch_alerts.csv",
                mime="text/csv",
            )

    # --- Alert timeline ---
    st.header("Alert Timeline")
    if comparison_mode:
        has_any_alerts = any(not adf.empty for _, adf, _ in comparison_alerts)
        if has_any_alerts:
            fig = create_alert_timeline(
                alerts_df,
                comparison_data=comparison_alerts,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No alerts triggered for the selected runs.")
    else:
        if alerts_df.empty:
            st.success("No alerts triggered for this run.")
        else:
            fig = create_alert_timeline(alerts_df, run_id=selected_run_id)
            st.plotly_chart(fig, use_container_width=True)

    # --- Alert history table ---
    st.header("Alert History")
    if all_alerts.empty:
        st.success("No alerts triggered.")
    else:
        display_cols = ["step", "detector", "severity", "message", "recommendation"]
        if comparison_mode and "run_id" in all_alerts.columns:
            display_cols = ["run_id"] + display_cols
        display_df = all_alerts[display_cols].copy()
        display_df.columns = [c.replace("_", " ").title() for c in display_cols]
        st.dataframe(display_df, use_container_width=True)

    # --- Run config (expandable) ---
    with st.expander("Run Configuration"):
        st.json(config)

    conn.close()

    # Auto-refresh: sleep THEN rerun. Must be at the end of the script so
    # all widgets have been rendered before the sleep blocks execution.
    if auto_refresh:
        time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    main()
