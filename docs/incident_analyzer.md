# IncidentAnalyzer Class Documentation

## Purpose and Functionality

The `IncidentAnalyzer` class is designed to analyze IT incident tickets using AI models. It provides functionality to determine the severity and type of incidents based on the text content of the tickets. The class uses pre-trained models from the `transformers` library to perform sentiment analysis and zero-shot classification.

## Initialization

The `IncidentAnalyzer` class is implemented as a singleton, ensuring that only one instance of the class is created. This is achieved using the `__new__` method. The class initializes the AI models on the available device (GPU if available, otherwise CPU).

### Code Example

```python
from ai_ticket_processing import IncidentAnalyzer

# Create an instance of IncidentAnalyzer
analyzer = IncidentAnalyzer()
```

## Methods

### `analyze_tickets_batch`

This method processes multiple tickets in batches. It takes a list of ticket texts and returns a list of dictionaries containing the severity and incident type for each ticket.

#### Parameters

- `texts` (List[str]): A list of ticket texts to be analyzed.

#### Returns

- `List[Dict[str, Any]]`: A list of dictionaries containing the analysis results for each ticket.

#### Code Example

```python
texts = [
    "The server is down and needs to be restarted.",
    "There is a bug in the login feature."
]

results = analyzer.analyze_tickets_batch(texts)
for result in results:
    print(result)
```

### `get_analyzer`

This function returns a cached instance of the `IncidentAnalyzer` class. It uses the `lru_cache` decorator to ensure that the instance is cached and reused.

#### Code Example

```python
from ai_ticket_processing import get_analyzer

# Get the cached instance of IncidentAnalyzer
analyzer = get_analyzer()
```

### `analyze_ticket_severity`

This function analyzes the severity of a single ticket text. It returns a float value representing the severity score.

#### Parameters

- `text` (str): The ticket text to be analyzed.

#### Returns

- `float`: The severity score of the ticket.

#### Code Example

```python
text = "The server is down and needs to be restarted."
severity = analyzer.analyze_ticket_severity(text)
print(f"Severity: {severity}")
```

### `extract_incident_patterns`

This function extracts the incident type and related types from a single ticket text. It returns a dictionary containing the incident type and related types.

#### Parameters

- `text` (str): The ticket text to be analyzed.

#### Returns

- `Dict[str, Any]`: A dictionary containing the incident type and related types.

#### Code Example

```python
text = "There is a bug in the login feature."
patterns = analyzer.extract_incident_patterns(text)
print(f"Incident Type: {patterns['incident_type']}")
print(f"Related Types: {patterns['related_types']}")
```
