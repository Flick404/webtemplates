#!/usr/bin/env python3
"""
FSVM Streamlit App - Complete Interface
"""

import streamlit as st
import time
import threading
import queue
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from datetime import datetime
from fsvm_core import FSVM

# Set page config
st.set_page_config(layout="wide", page_title="FSVM Dashboard")

# Initialize FSVM instance in session state
if 'fsvm' not in st.session_state:
    st.session_state.fsvm = FSVM()
    st.session_state.status_history = []
    st.session_state.chat_history = []

fsvm = st.session_state.fsvm

# Sidebar controls
with st.sidebar:
    st.title("🎛️ FSVM Controls")
    
    # Start/Stop controls
    if fsvm.is_running():
        if st.button("🛑 Stop FSVM", type="primary"):
            fsvm.stop()
            st.rerun()
    else:
        if st.button("▶️ Start FSVM", type="primary"):
            fsvm.start()
            st.rerun()
    
    st.write(f"**Status:** {'🟢 Running' if fsvm.is_running() else '🔴 Stopped'}")
    
    # Tension multiplier control
    current_multiplier = fsvm.get_tension_multiplier()
    new_multiplier = st.slider(
        "Tension Multiplier", 
        0.1, 5.0, 
        current_multiplier, 
        0.1,
        help="Adjust FSVM's sensitivity to tension"
    )
    if new_multiplier != current_multiplier:
        fsvm.set_tension_multiplier(new_multiplier)
        st.success(f"Tension multiplier set to {new_multiplier}")
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    if auto_refresh and fsvm.is_running():
        st.markdown("🔄 **Live updates enabled**")
    
    # Manual refresh
    if st.button("🔄 Manual Refresh"):
        st.rerun()

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "💬 Chat", "🔣 Symbols", "📜 Log"])

# Dashboard Tab
with tab1:
    st.title("🧠 FSVM Dashboard")
    
    # Get latest status
    status = fsvm.get_status_update()
    if status:
        st.session_state.status_history.append(status)
        if len(st.session_state.status_history) > 200:
            st.session_state.status_history.pop(0)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Cycle Count", fsvm.cycle_count)
    
    with col2:
        st.metric("Symbols", len(fsvm.tension_engine.symbols))
    
    with col3:
        current_tension = fsvm.tension_engine.get_current_tension()
        st.metric("Current Tension", f"{current_tension:.3f}")
    
    with col4:
        if hasattr(fsvm, 'start_time'):
            uptime = time.time() - fsvm.start_time
            st.metric("Uptime", f"{uptime:.1f}s")
        else:
            st.metric("Uptime", "N/A")
    
    # Tension graph
    if st.session_state.status_history:
        tensions = [s['current_tension'] for s in st.session_state.status_history]
        cycles = [s['cycle_count'] for s in st.session_state.status_history]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cycles, 
            y=tensions, 
            mode='lines', 
            name='Tension',
            line=dict(color='red', width=2)
        ))
        fig.update_layout(
            title="Tension Over Time",
            xaxis_title="Cognitive Cycles",
            yaxis_title="Tension Level",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Drives visualization
    if hasattr(fsvm, 'drives'):
        st.subheader("❤️‍🔥 Internal Drives")
        drives_df = pd.DataFrame(list(fsvm.drives.items()), columns=['Drive', 'Value'])
        fig = px.bar(drives_df, x='Drive', y='Value', 
                    title='FSVM Internal Drive States',
                    color='Value',
                    color_continuous_scale='RdYlBu')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent activity
    if hasattr(fsvm, 'activity_log') and fsvm.activity_log:
        st.subheader("📝 Recent Activity")
        for activity in fsvm.activity_log[-5:]:
            st.text(f"• {activity}")

# Chat Tab
with tab2:
    st.header("💬 Talk to FSVM")
    
    # Chat history
    if st.session_state.chat_history:
        st.subheader("Chat History")
        for msg in st.session_state.chat_history[-10:]:
            if msg['role'] == 'user':
                st.write(f"**You:** {msg['content']}")
            else:
                st.write(f"**FSVM:** {msg['content']}")
        st.markdown("---")
    
    # Chat input
    user_input = st.text_input("Your Message", key="chat_input")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Send", type="primary"):
            if user_input:
                fsvm.chat(user_input)
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': user_input,
                    'timestamp': time.time()
                })
                st.rerun()
    
    # Get response
    try:
        response = fsvm.chat_output_queue.get_nowait()
        if response:
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response,
                'timestamp': time.time()
            })
            st.success(f"**FSVM:** {response}")
    except queue.Empty:
        pass

# Symbols Tab
with tab3:
    st.header("🔣 Symbols")
    
    symbols = fsvm.tension_engine.symbols
    if symbols:
        # Symbol data
        data = []
        for sym in symbols.values():
            data.append({
                "ID": sym.id,
                "Freq": sym.frequency,
                "Last Seen": datetime.fromtimestamp(sym.last_seen).strftime('%H:%M:%S'),
                "Created": datetime.fromtimestamp(sym.created).strftime('%H:%M:%S'),
                "Pattern": " ".join(map(str, sym.pattern))[:50] + "..." if len(" ".join(map(str, sym.pattern))) > 50 else " ".join(map(str, sym.pattern)),
                "Type": "Meta" if sym.id.startswith('MS_') else "Regular",
                "Cluster": str(sym.cluster_id) if sym.cluster_id is not None else "None"
            })
        
        df = pd.DataFrame(data)
        
        # Sort by frequency
        df_sorted = df.sort_values("Freq", ascending=False)
        st.dataframe(df_sorted, use_container_width=True)
        
        # Symbol frequency chart
        st.subheader("Symbol Frequency Distribution")
        fig = px.bar(df_sorted.head(20), x='ID', y='Freq', 
                    title='Top 20 Symbols by Frequency')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Symbol type distribution
        st.subheader("Symbol Type Distribution")
        type_counts = df['Type'].value_counts()
        fig = px.pie(values=type_counts.values, names=type_counts.index, 
                    title='Symbol Types')
        st.plotly_chart(fig, use_container_width=True)
        
        # --- Symbol Relationship Graph ---
        st.subheader("Symbol Relationship Graph")
        G = fsvm.tension_engine.symbol_graph
        if G and G.number_of_nodes() > 0:
            pos = nx.spring_layout(G, seed=42)
            edge_x, edge_y = [], []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'),
                hoverinfo='none', mode='lines'
            )
            node_x, node_y = zip(*[pos[n] for n in G.nodes()])
            node_text = [str(symbols[n].pattern)[:30] for n in G.nodes()]
            node_trace = go.Scatter(
                x=node_x, y=node_y, mode='markers+text', hoverinfo='text',
                text=node_text, textposition='top center',
                marker=dict(showscale=False, color='blue', size=10)
            )
            fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
                title='Symbol Graph', showlegend=False, hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40)
            ))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No symbol graph to display yet.")
    else:
        st.info("No symbols created yet. Start the FSVM to see symbols emerge.")

# Log Tab
with tab4:
    st.header("📜 Activity Log")
    
    if hasattr(fsvm, 'activity_log') and fsvm.activity_log:
        # Show last 100 activities
        log_text = "\n".join(f"[{datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')}] {activity}" 
                            for activity in fsvm.activity_log[-100:])
        st.text_area("Recent Activity", log_text, height=400, disabled=True)
        
        # Activity summary
        st.subheader("Activity Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            total_activities = len(fsvm.activity_log)
            st.metric("Total Activities", total_activities)
            
            symbol_creations = len([a for a in fsvm.activity_log if "Created symbol" in a])
            st.metric("Symbol Creations", symbol_creations)
        
        with col2:
            meta_creations = len([a for a in fsvm.activity_log if "Created meta-symbol" in a])
            st.metric("Meta-Symbol Creations", meta_creations)
            
            recent_activities = len([a for a in fsvm.activity_log if time.time() - fsvm.start_time < 60])
            st.metric("Activities (Last Min)", recent_activities)
            
    else:
        st.info("No activity log available. Start the FSVM to see activity.")

# Auto-refresh logic
if auto_refresh and fsvm.is_running():
    time.sleep(2)
    st.rerun() 