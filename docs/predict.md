# Prediction Process Documentation

This document provides detailed information on the prediction process used in the project. It explains the purpose and functionality of the prediction process and includes instructions on how to configure and run the prediction process.

## Purpose and Functionality

The prediction process is designed to analyze IT tickets and predict their priority, category, and urgency. It leverages machine learning models to provide accurate predictions based on the ticket's summary, description, and other relevant features.

The main components of the prediction process are:
- **TicketPredictor**: A class that loads the trained model and performs predictions on individual tickets or batches of tickets.
- **predict_ticket_priority**: A convenience function for predicting the priority of a single ticket.
- **batch_predict_tickets**: A convenience function for predicting the priority of multiple tickets in a batch.

## Instructions

### Configuring the Prediction Process

1. **Model Path**: Ensure that the trained model is saved in the `models` directory with the filename `full_pipeline.joblib`. The `TicketPredictor` class will load this model for predictions.

2. **Dependencies**: Make sure all required dependencies are installed. You can find the list of dependencies in the `requirements.txt` file. Install them using the following command:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Prediction Process

#### Predicting a Single Ticket

To predict the priority, category, and urgency of a single ticket, use the `predict_ticket_priority` function. Here is an example:

```python
from predict import predict_ticket_priority

ticket = {
    "summary": "Server is down",
    "description": "The main server is not responding. We need to fix this ASAP.",
    "components": ["server"],
    "priority": "High"
}

prediction = predict_ticket_priority(ticket)
print(prediction)
```

#### Predicting Multiple Tickets

To predict the priority, category, and urgency of multiple tickets in a batch, use the `batch_predict_tickets` function. Here is an example:

```python
from predict import batch_predict_tickets

tickets = [
    {
        "summary": "Server is down",
        "description": "The main server is not responding. We need to fix this ASAP.",
        "components": ["server"],
        "priority": "High"
    },
    {
        "summary": "Feature request",
        "description": "We need a new feature to improve user experience.",
        "components": ["frontend"],
        "priority": "Medium"
    }
]

predictions = batch_predict_tickets(tickets)
for prediction in predictions:
    print(prediction)
```

### Interpreting the Results

The prediction results will include the following information:
- **predicted_priority**: The predicted priority of the ticket.
- **priority_confidence**: The confidence score for the predicted priority.
- **predicted_category**: The predicted category of the ticket.
- **category_confidence**: The confidence score for the predicted category.
- **urgency_score**: The urgency score of the ticket.
- **technical_indicators**: A dictionary of technical indicators and their respective confidence scores.

Use this information to prioritize and categorize IT tickets effectively.

## Conclusion

The prediction process is a crucial component of the project, enabling accurate and efficient analysis of IT tickets. By following the instructions provided in this document, you can configure and run the prediction process to obtain valuable insights into your IT tickets.
