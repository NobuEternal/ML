# AI Integration

## Purpose and Benefits

The AI integration in this project aims to automate the analysis and categorization of IT tickets. By leveraging machine learning models, the system can efficiently process large volumes of tickets, providing accurate severity analysis, incident pattern extraction, and priority prediction. The benefits of this AI integration include:

- **Improved Efficiency:** Automates the ticket analysis process, reducing the time and effort required for manual analysis.
- **Consistency:** Provides consistent and unbiased analysis of tickets, ensuring uniformity in ticket categorization.
- **Scalability:** Capable of handling large volumes of tickets, making it suitable for organizations with high ticket inflow.
- **Enhanced Decision Making:** Provides valuable insights and predictions that aid in decision-making and prioritization of tickets.

## Instructions for Using AI Features

### 1. IncidentAnalyzer Class

The `IncidentAnalyzer` class is responsible for analyzing ticket severity and extracting incident patterns. It uses pre-trained models from the `transformers` library.

#### Example Usage

```python
from ai_ticket_processing import IncidentAnalyzer

# Initialize the analyzer
analyzer = IncidentAnalyzer()

# Analyze a single ticket
text = "The server is down and causing a major outage."
severity = analyzer.analyze_ticket_severity(text)
patterns = analyzer.extract_incident_patterns(text)

print(f"Severity: {severity}")
print(f"Incident Type: {patterns['incident_type']}")
print(f"Related Types: {patterns['related_types']}")
```

### 2. TicketPredictor Class

The `TicketPredictor` class predicts the priority and category of tickets using a trained machine learning model.

#### Example Usage

```python
from predict import TicketPredictor

# Initialize the predictor
predictor = TicketPredictor()

# Predict the priority and category of a ticket
ticket_data = {
    "summary": "Server outage",
    "description": "The server is down and causing a major outage.",
    "components": ["server"],
    "priority": "High"
}

prediction = predictor.predict_ticket(ticket_data)

print(f"Predicted Priority: {prediction['predicted_priority']}")
print(f"Priority Confidence: {prediction['priority_confidence']}")
print(f"Predicted Category: {prediction['predicted_category']}")
print(f"Urgency Score: {prediction['urgency_score']}")
print(f"Technical Indicators: {prediction['technical_indicators']}")
```

### 3. Auto-Retraining

The `auto_retrain.py` script monitors the model's performance and triggers retraining when necessary. This ensures that the model remains accurate and up-to-date with the latest data.

#### Example Usage

```python
from auto_retrain import start_auto_retrain

# Start the auto-retraining process
start_auto_retrain()
```

### 4. API Endpoints

The API provides endpoints for predicting ticket priority and category. It uses FastAPI for building the API.

#### Example Usage

```python
import requests

# Define the API endpoint
url = "http://localhost:8000/predict"

# Define the ticket data
ticket_data = {
    "summary": "Server outage",
    "description": "The server is down and causing a major outage.",
    "components": ["server"],
    "priority": "High"
}

# Make a POST request to the API
response = requests.post(url, json=ticket_data)

# Print the prediction results
print(response.json())
```

By following these instructions, you can effectively utilize the AI features integrated into this project to analyze and categorize IT tickets.
