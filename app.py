import streamlit as st
import pandas as pd
import numpy as np
import io

# Try to import plotly, provide fallback if not available
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not available. Visualization features will be limited.")

# MRU Algorithm Implementation
def mru_page_replacement(page_reference, frame_count):
    """
    Simulate the MRU page replacement algorithm
    
    Args:
        page_reference: List of page references
        frame_count: Number of available frames
        
    Returns:
        Dictionary with simulation results
    """
    if not page_reference or frame_count <= 0:
        return {
            'frames_state': [],
            'page_faults': 0,
            'hits': 0,
            'hit_ratio': 0,
            'page_fault_rate': 0,
            'steps': []
        }
    
    frames = []
    frames_state = []
    steps = []
    page_faults = 0
    hits = 0
    
    # Initialize frames as empty
    frames = [-1] * frame_count  # -1 represents empty frame
    
    for step, page in enumerate(page_reference):
        step_info = {
            'step': step + 1,
            'page': page,
            'frames_before': frames.copy(),
            'status': '',
            'replaced_page': None
        }
        
        # Check if page is already in frames (hit)
        if page in frames:
            hits += 1
            step_info['status'] = 'Hit'
            # Update MRU position - move the page to the end of the list
            frames.remove(page)
            frames.append(page)
        else:
            # Page fault
            page_faults += 1
            step_info['status'] = 'Fault'
            
            # If there's an empty frame
            if -1 in frames:
                # Replace the first empty frame
                empty_index = frames.index(-1)
                step_info['replaced_page'] = 'Empty'
                frames[empty_index] = page
                # Move to end to mark as most recently used
                frames.remove(page)
                frames.append(page)
            else:
                # MRU: Remove the most recently used page (last in the list)
                replaced = frames.pop()
                step_info['replaced_page'] = replaced
                frames.append(page)
        
        step_info['frames_after'] = frames.copy()
        steps.append(step_info)
        frames_state.append(frames.copy())
    
    total_references = len(page_reference)
    hit_ratio = (hits / total_references) * 100 if total_references > 0 else 0
    page_fault_rate = (page_faults / total_references) * 100 if total_references > 0 else 0
    
    return {
        'frames_state': frames_state,
        'page_faults': page_faults,
        'hits': hits,
        'hit_ratio': hit_ratio,
        'page_fault_rate': page_fault_rate,
        'steps': steps
    }

# Simple table visualization (fallback when plotly is not available)
def create_simple_table(frames_state, page_reference):
    """Create a simple HTML table for visualization"""
    if not frames_state:
        return "<p>No data to display</p>"
    
    frame_count = len(frames_state[0])
    html = "<table style='border-collapse: collapse; width: 100%;'>"
    
    # Header
    html += "<tr><th style='border: 1px solid; padding: 8px;'>Step</th>"
    html += "<th style='border: 1px solid; padding: 8px;'>Page</th>"
    for i in range(frame_count):
        html += f"<th style='border: 1px solid; padding: 8px;'>Frame {i+1}</th>"
    html += "</tr>"
    
    # Data rows
    for i, (state, page) in enumerate(zip(frames_state, page_reference)):
        html += f"<tr><td style='border: 1px solid; padding: 8px;'>{i+1}</td>"
        html += f"<td style='border: 1px solid; padding: 8px; font-weight: bold;'>{page}</td>"
        for frame_val in state:
            color = 'lightgreen' if frame_val != -1 else 'lightgray'
            html += f"<td style='border: 1px solid; padding: 8px; background-color: {color};'>{frame_val if frame_val != -1 else ''}</td>"
        html += "</tr>"
    
    html += "</table>"
    return html

# Plotly visualization (if available)
def create_plotly_visualization(frames_state, page_reference):
    """Create visualization using Plotly"""
    if not PLOTLY_AVAILABLE or not frames_state:
        return None
    
    frame_count = len(frames_state[0])
    
    # Create DataFrame for visualization
    df = pd.DataFrame()
    for i in range(frame_count):
        df[f"Frame {i+1}"] = [state[i] if state[i] != -1 else None for state in frames_state]
    df["Page Reference"] = page_reference
    
    # Create figure
    fig = go.Figure()
    
    # Add traces for each frame
    for i in range(frame_count):
        fig.add_trace(go.Scatter(
            x=list(range(len(frames_state))),
            y=[i+1] * len(frames_state),
            mode='markers+text',
            marker=dict(
                size=40,
                color=[state[i] if state[i] != -1 else 'lightgray' for state in frames_state],
                colorscale='Viridis',
                showscale=True if i == 0 else False,
                cmin=0,
                cmax=max(max(page_reference), frame_count) if page_reference else frame_count,
                colorbar=dict(title="Page Number") if i == 0 else None
            ),
            text=[state[i] if state[i] != -1 else "" for state in frames_state],
            textfont=dict(color="white" if state[i] != -1 else "black", size=10),
            name=f"Frame {i+1}"
        ))
    
    # Add page reference annotations
    for i, page in enumerate(page_reference):
        fig.add_annotation(
            x=i,
            y=0,
            text=str(page),
            showarrow=False,
            font=dict(color="black", size=12, weight="bold")
        )
    
    fig.update_layout(
        title="MRU Page Replacement Process",
        xaxis_title="Step",
        yaxis_title="Frame",
        yaxis=dict(range=[0, frame_count+1], autorange=False),
        height=400,
        hovermode="closest"
    )
    
    return fig

# Main application
def main():
    st.set_page_config(page_title="MRU Page Replacement Simulator", layout="wide")
    
    st.title("🔄 MRU (Most Recently Used) Page Replacement Algorithm")
    st.markdown("""
    Simulate and analyze the MRU page replacement algorithm with interactive visualizations.
    """)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Frame count input
    frame_count = st.sidebar.slider(
        "Number of Frames", 
        min_value=3, 
        max_value=8, 
        value=4,
        help="Select the number of available frames (3-8)"
    )
    
    # Input method selection
    input_method = st.sidebar.radio(
        "📥 Input Method", 
        ["Manual Input", "Upload File"],
        help="Choose how to provide page references"
    )
    
    page_reference = []
    
    # Input handling
    if input_method == "Upload File":
        st.subheader("📁 Upload File")
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file", 
            type=["csv", "xlsx", "xls"],
            help="Upload a file containing page references in the first column"
        )
        
        if uploaded_file is not None:
            try:
                # Read file based on type
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success("✅ File uploaded successfully!")
                st.dataframe(df.head(10))  # Show first 10 rows
                
                # Extract page references from first column
                if len(df.columns) > 0:
                    raw_data = df.iloc[:, 0].tolist()
                    page_reference = []
                    
                    for item in raw_data:
                        try:
                            if pd.notna(item):
                                page_reference.append(int(item))
                        except (ValueError, TypeError):
                            continue
                    
                    # Limit to 20 pages
                    if len(page_reference) > 20:
                        st.warning(f"⚠️ Limiting to first 20 page references (found {len(page_reference)})")
                        page_reference = page_reference[:20]
                    
                    if page_reference:
                        st.write(f"📊 Extracted {len(page_reference)} page references:")
                        st.code(", ".join(map(str, page_reference)))
                    else:
                        st.error("❌ No valid page references found in the file")
                        
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.info("Please ensure your file has numeric page references in the first column")
    
    else:  # Manual Input
        st.subheader("✏️ Manual Input")
        default_input = "7,0,1,2,0,3,0,4,2,3,0,3,2,1,2,0,1,7,0,1"
        input_text = st.text_area(
            "Enter page references (comma-separated)", 
            value=default_input,
            help="Enter page numbers separated by commas (e.g., 1,2,3,4,5)"
        )
        
        if input_text:
            try:
                page_reference = []
                for item in input_text.split(","):
                    item = item.strip()
                    if item:
                        page_reference.append(int(item))
                
                # Limit to 20 pages
                if len(page_reference) > 20:
                    st.warning(f"⚠️ Limiting to first 20 page references (entered {len(page_reference)})")
                    page_reference = page_reference[:20]
                
                if page_reference:
                    st.write(f"📊 Page references ({len(page_reference)}):")
                    st.code(", ".join(map(str, page_reference)))
                    
            except ValueError:
                st.error("❌ Invalid input. Please enter comma-separated integers only.")
                page_reference = []
    
    # Run simulation
    if page_reference and frame_count:
        st.header("📈 Simulation Results")
        
        # Run MRU algorithm
        results = mru_page_replacement(page_reference, frame_count)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📄 Total References", len(page_reference))
        col2.metric("❌ Page Faults", results['page_faults'])
        col3.metric("✅ Hits", results['hits'])
        col4.metric("📊 Hit Ratio", f"{results['hit_ratio']:.2f}%")
        
        st.metric("🔴 Page Fault Rate", f"{results['page_fault_rate']:.2f}%")
        
        # Visualization
        st.subheader("🎨 Visualization")
        
        if PLOTLY_AVAILABLE:
            fig = create_plotly_visualization(results['frames_state'], page_reference)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No visualization data available")
        else:
            st.markdown("### Frame State Table")
            st.markdown(create_simple_table(results['frames_state'], page_reference), unsafe_allow_html=True)
        
        # Step-by-step execution
        st.subheader("📋 Step-by-Step Execution")
        
        # Create detailed DataFrame
        step_data = []
        for step in results['steps']:
            row = {
                'Step': step['step'],
                'Page': step['page'],
                'Status': step['status'],
                'Replaced': step['replaced_page'] if step['replaced_page'] is not None else '-'
            }
            for i in range(frame_count):
                frame_val = step['frames_after'][i]
                row[f'Frame {i+1}'] = frame_val if frame_val != -1 else ''
            step_data.append(row)
        
        step_df = pd.DataFrame(step_data)
        st.dataframe(step_df, use_container_width=True)
        
        # Download report
        st.subheader("💾 Export Results")
        if st.button("📥 Generate CSV Report"):
            # Create report DataFrame
            report_data = {
                'Metric': [
                    'Total Page References',
                    'Number of Frames',
                    'Page Faults',
                    'Hits',
                    'Hit Ratio (%)',
                    'Page Fault Rate (%)'
                ],
                'Value': [
                    len(page_reference),
                    frame_count,
                    results['page_faults'],
                    results['hits'],
                    f"{results['hit_ratio']:.2f}",
                    f"{results['page_fault_rate']:.2f}"
                ]
            }
            report_df = pd.DataFrame(report_data)
            
            # Convert to CSV
            csv = report_df.to_csv(index=False)
            
            # Add step-by-step data
            csv += "\n\nStep-by-Step Execution\n"
            csv += step_df.to_csv(index=False)
            
            st.download_button(
                label="📊 Download Report",
                data=csv,
                file_name=f"mru_simulation_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Information section
    st.markdown("---")
    st.header("ℹ️ About MRU Algorithm")
    
    with st.expander("📖 Learn about MRU"):
        st.markdown("""
        ### Most Recently Used (MRU) Page Replacement
        
        The **MRU algorithm** replaces the page that was most recently used when a page fault occurs.
        
        **How it works:**
        1. When a page is accessed, it's marked as most recently used
        2. When a page fault occurs and all frames are full, the most recently used page is replaced
        3. This is the opposite of LRU (Least Recently Used)
        
        **Characteristics:**
        - Performs well in certain loop-based access patterns
        - May perform poorly for sequential access patterns
        - Simple to implement with a stack or list structure
        
        **When to use MRU:**
        - When the most recently accessed page is least likely to be needed again
        - In scenarios with cyclic access patterns
        - When there's a temporal locality that favors older pages
        """)

if __name__ == "__main__":
    main()