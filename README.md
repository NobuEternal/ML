# AI Ticket Processing System

This project is designed to process IT tickets using AI models to analyze ticket severity, extract incident patterns, and predict ticket priority and category. The system integrates various AI models and provides an API for easy integration with other tools.

## Table of Contents
- [Overview](#overview)
- [Setup and Installation](#setup-and-installation)
- [Running the Project](#running-the-project)
- [AI Integration](#ai-integration)
- [Documentation](#documentation)

## Overview
The AI Ticket Processing System leverages machine learning models to automate the analysis and categorization of IT tickets. It includes features such as:
- Severity analysis
- Incident pattern extraction
- Priority and category prediction
- Auto-retraining of models
- Monitoring and dashboard

## Setup and Installation
To set up and run the project, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/NobuEternal/ML.git
   cd ML
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up MongoDB:**
   Ensure you have MongoDB installed and running. Update the `config.py` file with your MongoDB URI and database details.

5. **Run the CUDA test script (optional):**
   ```bash
   python cuda_test.py
   ```

## Running the Project
To run the project, follow these steps:

1. **Start the data extraction process:**
   ```bash
   python data_extraction.py
   ```

2. **Run the data preprocessing script:**
   ```bash
   python data_preprocessing.py
   ```

3. **Train the model:**
   ```bash
   python model_training.py
   ```

4. **Start the API server:**
   ```bash
   python api.py
   ```

5. **Open the dashboard:**
   ```bash
   streamlit run dashboard.py
   ```

## AI Integration
The AI integration in this project involves several components:

1. **IncidentAnalyzer Class:**
   The `IncidentAnalyzer` class is responsible for analyzing ticket severity and extracting incident patterns. It uses pre-trained models from the `transformers` library.

2. **TicketPredictor Class:**
   The `TicketPredictor` class predicts the priority and category of tickets using a trained machine learning model.

3. **Auto-Retraining:**
   The `auto_retrain.py` script monitors the model's performance and triggers retraining when necessary.

4. **API Endpoints:**
   The API provides endpoints for predicting ticket priority and category. It uses FastAPI for building the API.

For detailed documentation on each component, refer to the `docs` folder.

## Documentation
The project includes detailed documentation for each component:

- [AI Integration](docs/ai_integration.md)
- [IncidentAnalyzer](docs/incident_analyzer.md)
- [API Endpoints](docs/api.md)
- [Auto-Retraining](docs/auto_retrain.md)
- [Configuration](docs/config.md)
- [CUDA Test](docs/cuda_test.md)
- [Dashboard](docs/dashboard.md)
- [Data Extraction](docs/data_extraction.md)
- [Data Preprocessing](docs/data_preprocessing.md)
- [Field Configuration](docs/field_config.md)
- [GUI](docs/gui.md)
- [Main Script](docs/main.md)
- [Model Training](docs/model_training.md)
- [Monitoring](docs/monitoring.md)
- [Prediction](docs/predict.md)
- [Project Requirements](docs/requirements.md)
- [Test Fetch](docs/test_fetch.md)
