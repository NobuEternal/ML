# Dashboard

## Purpose
The dashboard provides a user-friendly interface to visualize the performance of the AI Ticket Processing System and to test predictions on new tickets.

## Functionality
The dashboard includes the following features:
- Display model performance metrics
- Visualize accuracy trends over time
- Provide an interface for testing ticket predictions

## Instructions

### Running the Dashboard
To run the dashboard, use the following command:
```bash
streamlit run dashboard.py
```

### Using the Dashboard
1. **Model Performance Metrics:**
   - The dashboard displays key performance metrics such as overall accuracy, average precision, and average recall.
   - A line chart visualizes the accuracy trend over time.

2. **Test Prediction:**
   - Enter the ticket summary and description in the provided input fields.
   - Click the "Predict" button to get the predicted priority, category, and other relevant information for the ticket.

### Interpreting the Results
- **Overall Accuracy:** Indicates the percentage of correct predictions made by the model.
- **Average Precision:** Measures the accuracy of the positive predictions.
- **Average Recall:** Measures the ability of the model to identify all relevant instances.
- **Accuracy Trend:** Shows how the model's accuracy has changed over time.
- **Predicted Priority:** The predicted priority level of the ticket.
- **Predicted Category:** The predicted category of the ticket.
- **Technical Indicators:** Additional technical information extracted from the ticket.

By following these instructions, you can effectively use the dashboard to monitor the performance of the AI Ticket Processing System and test predictions on new tickets.
