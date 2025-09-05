#!/usr/bin/env python3
"""
Streamlit dashboard for viewing Standard Agent evaluation results.

Usage:
    streamlit run evaluation/dashboard.py

Features:
- Browse evaluation runs from JSONL files
- View detailed spans and traces
- Filter by success/failure, time ranges, token usage
- Aggregate metrics and charts
"""

import json
import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Standard Agent Evaluation Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of records."""
    if not file_path.exists():
        return []
    
    records = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records

def load_runs_data(eval_runs_dir: Path) -> pd.DataFrame:
    """Load all evaluation runs from the eval_runs directory."""
    all_runs = []
    
    for jsonl_file in eval_runs_dir.glob("*.jsonl"):
        if jsonl_file.name == "spans.jsonl":
            continue  # Skip spans file
        
        records = load_jsonl(jsonl_file)
        for record in records:
            record["source_file"] = jsonl_file.name
            all_runs.append(record)
    
    if not all_runs:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_runs)
    
    # Convert timestamp to datetime
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    
    return df

def load_spans_data(eval_runs_dir: Path) -> pd.DataFrame:
    """Load spans data."""
    spans_file = eval_runs_dir / "spans.jsonl"
    spans = load_jsonl(spans_file)
    
    if not spans:
        return pd.DataFrame()
    
    df = pd.DataFrame(spans)
    
    # Convert timestamps
    if "start_time" in df.columns:
        df["start_time"] = pd.to_datetime(df["start_time"])
    if "end_time" in df.columns:
        df["end_time"] = pd.to_datetime(df["end_time"])
    
    return df

def format_duration(ms: Optional[float]) -> str:
    """Format duration in milliseconds to human readable."""
    if ms is None or pd.isna(ms):
        return "N/A"
    
    if ms < 1000:
        return f"{ms:.0f}ms"
    elif ms < 60000:
        return f"{ms/1000:.1f}s"
    else:
        return f"{ms/60000:.1f}min"

def format_tokens(tokens: Optional[int]) -> str:
    """Format token count."""
    if tokens is None or pd.isna(tokens):
        return "N/A"
    return f"{tokens:,}"

def main():
    st.title("🤖 Standard Agent Evaluation Dashboard")
    
    # Sidebar for file selection
    st.sidebar.header("📁 Data Source")
    
    eval_runs_dir = Path("./eval_runs")
    if not eval_runs_dir.exists():
        st.error(f"Evaluation runs directory not found: {eval_runs_dir}")
        st.info("Run some evaluations first using: `python -m evaluation.runner run ...`")
        return
    
    # Load data
    runs_df = load_runs_data(eval_runs_dir)
    spans_df = load_spans_data(eval_runs_dir)
    
    if runs_df.empty:
        st.warning("No evaluation runs found in ./eval_runs/")
        st.info("Run some evaluations first using: `python -m evaluation.runner run ...`")
        return
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Success filter
    success_filter = st.sidebar.selectbox(
        "Success Status",
        ["All", "Success Only", "Failures Only"]
    )
    
    # Agent filter
    if "agent_name" in runs_df.columns:
        agents = ["All"] + sorted(runs_df["agent_name"].dropna().unique().tolist())
        agent_filter = st.sidebar.selectbox("Agent", agents)
    else:
        agent_filter = "All"
    
    # Dataset filter
    if "dataset_id" in runs_df.columns:
        datasets = ["All"] + sorted(runs_df["dataset_id"].dropna().unique().tolist())
        dataset_filter = st.sidebar.selectbox("Dataset", datasets)
    else:
        dataset_filter = "All"
    
    # Apply filters
    filtered_df = runs_df.copy()
    
    if success_filter == "Success Only":
        filtered_df = filtered_df[filtered_df["success"] == True]
    elif success_filter == "Failures Only":
        filtered_df = filtered_df[filtered_df["success"] == False]
    
    if agent_filter != "All":
        filtered_df = filtered_df[filtered_df["agent_name"] == agent_filter]
    
    if dataset_filter != "All":
        filtered_df = filtered_df[filtered_df["dataset_id"] == dataset_filter]
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Runs", len(filtered_df))
    
    with col2:
        success_rate = (filtered_df["success"] == True).mean() * 100 if len(filtered_df) > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with col3:
        avg_time = filtered_df["time_ms"].mean() if "time_ms" in filtered_df.columns else None
        st.metric("Avg Duration", format_duration(avg_time))
    
    with col4:
        total_tokens = filtered_df["tokens_total"].sum() if "tokens_total" in filtered_df.columns else None
        st.metric("Total Tokens", format_tokens(total_tokens))
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📋 Runs Detail", "🔍 Spans", "📈 Charts"])
    
    with tab1:
        st.header("Overview")
        
        if len(filtered_df) > 0:
            # Success/Failure pie chart
            col1, col2 = st.columns(2)
            
            with col1:
                success_counts = filtered_df["success"].value_counts()
                fig_pie = px.pie(
                    values=success_counts.values,
                    names=["Success" if x else "Failure" for x in success_counts.index],
                    title="Success vs Failure"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Duration histogram
                if "time_ms" in filtered_df.columns:
                    fig_hist = px.histogram(
                        filtered_df,
                        x="time_ms",
                        title="Duration Distribution",
                        labels={"time_ms": "Duration (ms)"}
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
    
    with tab2:
        st.header("Evaluation Runs")
        
        if len(filtered_df) > 0:
            # Display columns selection
            display_cols = ["run_id", "timestamp_utc", "goal", "success", "time_ms", "tokens_total", "agent_name", "dataset_id"]
            available_cols = [col for col in display_cols if col in filtered_df.columns]
            
            display_df = filtered_df[available_cols].copy()
            
            # Format columns
            if "timestamp_utc" in display_df.columns:
                display_df["timestamp_utc"] = display_df["timestamp_utc"].dt.strftime("%Y-%m-%d %H:%M:%S")
            
            if "time_ms" in display_df.columns:
                display_df["duration"] = display_df["time_ms"].apply(format_duration)
            
            if "tokens_total" in display_df.columns:
                display_df["tokens"] = display_df["tokens_total"].apply(format_tokens)
            
            if "goal" in display_df.columns:
                display_df["goal_preview"] = display_df["goal"].apply(
                    lambda x: x[:100] + "..." if isinstance(x, str) and len(x) > 100 else x
                )
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Run details expander
            st.subheader("Run Details")
            run_ids = filtered_df["run_id"].tolist()
            selected_run = st.selectbox("Select a run to view details:", run_ids)
            
            if selected_run:
                run_data = filtered_df[filtered_df["run_id"] == selected_run].iloc[0]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Goal:**")
                    st.text_area("", value=run_data.get("goal", "N/A"), height=100, disabled=True, key="goal")
                
                with col2:
                    st.write("**Result:**")
                    result_text = str(run_data.get("result", "N/A"))
                    st.text_area("", value=result_text, height=100, disabled=True, key="result")
                
                # Full run data as JSON
                with st.expander("Full Run Data (JSON)"):
                    st.json(run_data.to_dict())
    
    with tab3:
        st.header("Spans & Traces")
        
        if not spans_df.empty:
            # Filter spans by selected run
            if "run_id" in spans_df.columns:
                run_ids = ["All"] + sorted(spans_df["run_id"].unique().tolist())
                selected_run_span = st.selectbox("Filter by Run ID:", run_ids, key="span_run_filter")
                
                span_filtered = spans_df.copy()
                if selected_run_span != "All":
                    span_filtered = span_filtered[span_filtered["run_id"] == selected_run_span]
                
                st.dataframe(span_filtered, use_container_width=True, hide_index=True)
            else:
                st.dataframe(spans_df, use_container_width=True, hide_index=True)
        else:
            st.info("No spans data found. Make sure spans.jsonl exists in ./eval_runs/")
    
    with tab4:
        st.header("Charts & Analytics")
        
        if len(filtered_df) > 0:
            # Time series of runs
            if "timestamp_utc" in filtered_df.columns:
                fig_timeline = px.scatter(
                    filtered_df,
                    x="timestamp_utc",
                    y="time_ms",
                    color="success",
                    size="tokens_total" if "tokens_total" in filtered_df.columns else None,
                    title="Runs Over Time",
                    labels={"time_ms": "Duration (ms)", "timestamp_utc": "Time"}
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Token usage vs duration
            if "tokens_total" in filtered_df.columns and "time_ms" in filtered_df.columns:
                fig_tokens = px.scatter(
                    filtered_df,
                    x="tokens_total",
                    y="time_ms",
                    color="success",
                    title="Token Usage vs Duration",
                    labels={"tokens_total": "Total Tokens", "time_ms": "Duration (ms)"}
                )
                st.plotly_chart(fig_tokens, use_container_width=True)

if __name__ == "__main__":
    main()
