import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import time

# MRU Algorithm Implementation
def mru_page_replacement(page_reference, frame_count):
    """
    Simulate the MRU page replacement algorithm
    
    Args:
        page_reference: List of page references
        frame_count: Number of available frames
        
    Returns:
        Tuple of (frames_state, page_faults, hit_ratio)
        frames_state: List of lists showing frame contents at each step
        page_faults: Total number of page faults
        hit_ratio: Hit ratio percentage
    """
    frames = []
    frames_state = []
    page_faults = 0
    hits = 0
    
    # Initialize frames as empty
    for _ in range(frame_count):
        frames.append(-1)  # -1 represents empty frame
    
    for page in page_reference:
        # Check if page is already in frames (hit)
        if page in frames:
            hits += 1
            # Update MRU position - move the page to the end of the list
            frames.remove(page)
            frames.append(page)
        else:
            # Page fault
            page_faults += 1
            
            # If there's an empty frame
            if -1 in frames:
                # Replace the first empty frame
                empty_index = frames.index(-1)
                frames[empty_index] = page
                # Move to end to mark as most recently used
                frames.remove(page)
                frames.append(page)
            else:
                # MRU: Remove the most recently used page (last in the list)
                frames.pop()
                frames.append(page)
        
        # Store current state of frames
        frames_state.append(frames.copy())
    
    hit_ratio = (hits / len(page_reference)) * 100 if page_reference else 0
    return frames_state, page_faults, hit_ratio

# Function to visualize page replacement process
def visualize_page_replacement(frames_state, page_reference):
    """
    Create a visualization of the page replacement process
    
    Args:
        frames_state: List of lists showing frame contents at each step
        page_reference: List of page references
        
    Returns:
        Plotly figure
    """
    # Create a DataFrame for visualization
    frame_count = len(frames_state[0]) if frames_state else 0
    df = pd.DataFrame()
    
    for i in range(frame_count):
        column_name = f"Frame {i+1}"
        df[column_name] = [state[i] for state in frames_state]
    
    df["Page Reference"] = page_reference
    
    # Create a heatmap-like visualization
    fig = go.Figure()
    
    for i in range(frame_count):
        fig.add_trace(go.Scatter(
            x=list(range(len(frames_state))),
            y=[i+1] * len(frames_state),
            mode='markers',
            marker=dict(
                size=30,
                color=[state[i] for state in frames_state],
                colorscale='Viridis',
                showscale=True,
                cmin=0,
                cmax=max(max(page_reference), frame_count) if page_reference else frame_count,
                colorbar=dict(title="Page Number")
            ),
            text=[state[i] if state[i] != -1 else "" for state in frames_state],
            textfont=dict(color="white", size=10),
            name=f"Frame {i+1}"
        ))
    
    # Add page reference as annotations
    for i, page in enumerate(page_reference):
        fig.add_annotation(
            x=i,
            y=0,
            text=str(page),
            showarrow=False,
            font=dict(color="black", size=12)
        )
    
    fig.update_layout(
        title="MRU Page Replacement Process",
        xaxis_title="Step",
        yaxis_title="Frame",
        yaxis=dict(range=[0, frame_count+1], autorange=False),
        height=400
    )
    
    return fig

# Main Streamlit application
def main():
    st.title("MRU (Most Recently Used) Page Replacement Algorithm Simulator")
    st.markdown("""
    This application simulates the MRU page replacement algorithm and calculates the page fault rate.
    You can upload a CSV or Excel file containing page references or manually input them.
    """)
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Frame count input
    frame_count = st.sidebar.slider("Number of Frames", min_value=3, max_value=8, value=4)
    
    # Page reference input method
    input_method = st.sidebar.radio("Input Method", ["Upload File", "Manual Input"])
    
    page_reference = []
    
    if input_method == "Upload File":
        st.subheader("Upload File")
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success("File uploaded successfully!")
                st.dataframe(df)
                
                # Try to extract page references from the file
                # Assume the first column contains page references
                if len(df.columns) > 0:
                    page_reference = df.iloc[:, 0].tolist()
                    # Remove any NaN values
                    page_reference = [int(p) for p in page_reference if pd.notna(p) and str(p).isdigit()]
                    
                    # Limit to 20 pages as per requirement
                    if len(page_reference) > 20:
                        st.warning(f"Limiting to first 20 page references out of {len(page_reference)}")
                        page_reference = page_reference[:20]
                    
                    st.write(f"Extracted {len(page_reference)} page references:")
                    st.write(page_reference)
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    else:
        st.subheader("Manual Input")
        input_text = st.text_area("Enter page references (comma-separated)", "7,0,1,2,0,3,0,4,2,3,0,3,2,1,2,0,1,7,0,1")
        
        if input_text:
            try:
                page_reference = [int(p.strip()) for p in input_text.split(",")]
                # Limit to 20 pages as per requirement
                if len(page_reference) > 20:
                    st.warning(f"Limiting to first 20 page references out of {len(page_reference)}")
                    page_reference = page_reference[:20]
                
                st.write(f"Page references: {page_reference}")
            except ValueError:
                st.error("Invalid input. Please enter comma-separated integers.")
    
    # Run simulation if we have page references
    if page_reference and frame_count:
        st.header("Simulation Results")
        
        # Run MRU algorithm
        frames_state, page_faults, hit_ratio = mru_page_replacement(page_reference, frame_count)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Page References", len(page_reference))
        col2.metric("Page Faults", page_faults)
        col3.metric("Hit Ratio", f"{hit_ratio:.2f}%")
        
        # Calculate page fault rate
        page_fault_rate = (page_faults / len(page_reference)) * 100
        st.metric("Page Fault Rate", f"{page_fault_rate:.2f}%")
        
        # Visualize page replacement process
        st.subheader("Page Replacement Process Visualization")
        fig = visualize_page_replacement(frames_state, page_reference)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show step-by-step execution
        st.subheader("Step-by-Step Execution")
        
        # Create a DataFrame for the step-by-step table
        step_df = pd.DataFrame()
        step_df["Step"] = range(1, len(page_reference) + 1)
        step_df["Page Reference"] = page_reference
        
        for i in range(frame_count):
            step_df[f"Frame {i+1}"] = [state[i] if state[i] != -1 else "" for state in frames_state]
        
        # Add status column (Hit or Fault)
        status = []
        for i, page in enumerate(page_reference):
            if i > 0 and page in frames_state[i-1]:
                status.append("Hit")
            else:
                status.append("Fault")
        step_df["Status"] = status
        
        st.dataframe(step_df, use_container_width=True)
        
        # Generate downloadable report
        st.subheader("Generate Report")
        if st.button("Generate Report"):
            report = pd.DataFrame({
                "Metric": ["Total Page References", "Number of Frames", "Page Faults", "Hit Ratio", "Page Fault Rate"],
                "Value": [len(page_reference), frame_count, page_faults, f"{hit_ratio:.2f}%", f"{page_fault_rate:.2f}%"]
            })
            
            csv = report.to_csv(index=False)
            st.download_button(
                label="Download Report as CSV",
                data=csv,
                file_name="mru_simulation_report.csv",
                mime="text/csv"
            )

# Additional information section
def about_section():
    st.header("About MRU Algorithm")
    st.markdown("""
    The Most Recently Used (MRU) page replacement algorithm is the opposite of the Least Recently Used (LRU) algorithm.
    
    In MRU, when a page needs to be replaced, the algorithm selects the page that was most recently used.
    This algorithm is based on the principle that if a page was heavily used recently, it's less likely to be needed again soon.
    
    **Key Points:**
    - MRU replaces the most recently used page when a page fault occurs
    - It performs well in certain scenarios where there's a loop in the page references
    - It's not as commonly used as LRU but can be effective in specific situations
    """)

# Run the app
if __name__ == "__main__":
    main()
    
    # Add about section
    st.markdown("---")
    about_section()