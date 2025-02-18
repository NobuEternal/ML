import streamlit as st
import pandas as pd
import plotly.graph_objects as go  # Change from plotly.express to plotly.graph_objects
from predict import TicketPredictor
from monitoring import ModelMonitor
import json

def main():
    st.title("IT Ticket Processing Dashboard")
    
    # Model Performance Metrics
    st.header("Model Performance")
    try:
        with open("reports/performance_report.json") as f:
            metrics = json.load(f)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Overall Accuracy", f"{metrics['accuracy']:.2f}")
        col2.metric("Average Precision", f"{metrics['weighted avg']['precision']:.2f}")
        col3.metric("Average Recall", f"{metrics['weighted avg']['recall']:.2f}")
        
        # Use plotly.graph_objects instead of plotly.express
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=metrics['accuracy_history'],
            mode='lines',
            name='Accuracy'
        ))
        st.plotly_chart(fig)
        
    except Exception as e:
        st.error(f"Error loading metrics: {str(e)}")
    
    # Prediction Interface
    st.header("Test Prediction")
    summary = st.text_input("Ticket Summary")
    description = st.text_area("Ticket Description")
    
    if st.button("Predict"):
        predictor = TicketPredictor()
        result = predictor.predict_ticket({
            "summary": summary,
            "description": description
        })
        st.json(result)

if __name__ == "__main__":
    main()
