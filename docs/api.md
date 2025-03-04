# API Endpoints Documentation

## Purpose and Functionality

The API provides endpoints for predicting the priority and category of IT tickets. It uses FastAPI to build the API and integrates with the `TicketPredictor` class to perform predictions.

## Endpoints

### `/predict`

This endpoint accepts a ticket object and returns the predicted priority, category, and other related information.

#### Method

- `POST`

#### Request Body

The request body should be a JSON object representing the ticket. The following fields are required:

- `summary` (str): A brief summary of the ticket.
- `description` (str): A detailed description of the ticket.
- `components` (Optional[List[str]]): A list of components related to the ticket.
- `priority` (Optional[str]): The priority of the ticket.

#### Response

The response will be a JSON object containing the predicted priority, category, and other related information. The following fields are included:

- `predicted_priority` (str): The predicted priority of the ticket.
- `priority_confidence` (float): The confidence score for the predicted priority.
- `predicted_category` (str): The predicted category of the ticket.
- `technical_indicators` (Dict[str, float]): A dictionary of technical indicators and their confidence scores.
- `urgency_score` (float): The urgency score of the ticket.

#### Example Request

```json
{
  "summary": "Server outage",
  "description": "The server is down and causing a major outage.",
  "components": ["server"],
  "priority": "High"
}
```

#### Example Response

```json
{
  "predicted_priority": "Critical",
  "priority_confidence": 0.95,
  "predicted_category": "Infrastructure",
  "technical_indicators": {
    "Infrastructure": 0.95,
    "Security": 0.05
  },
  "urgency_score": 9.5
}
```

## Usage Instructions

To use the API, follow these steps:

1. **Start the API server:**
   ```bash
   python api.py
   ```

2. **Make a POST request to the `/predict` endpoint:**
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

By following these instructions, you can effectively use the API to predict the priority and category of IT tickets.
