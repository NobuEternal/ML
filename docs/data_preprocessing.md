# Data Preprocessing Process

This document provides detailed information on the data preprocessing process used in the AI Ticket Processing System.

## Purpose and Functionality

The data preprocessing process is responsible for cleaning and transforming the raw IT ticket data into a format suitable for machine learning model training and prediction. This includes handling missing values, extracting relevant features, and normalizing the data.

## Configuration

Before running the data preprocessing process, ensure that the necessary dependencies are installed and the configuration settings are properly set up.

## Running the Data Preprocessing Process

To run the data preprocessing process, execute the `data_preprocessing.py` script. This script will read the raw ticket data, apply various preprocessing steps, and save the cleaned data to a file.

Example command:
```bash
python data_preprocessing.py
```

## Functions

The `data_preprocessing.py` script includes the following functions:

### `safe_get(data: Any, path: str, default: Any = None) -> Any`

Robust nested data access with type checking.

### `process_datetime(value: Any) -> Optional[datetime]`

Safe datetime parsing with correct type hints.

### `process_components(components: List[Dict]) -> str`

Process components with proper error handling.

### `process_labels(labels: List[str]) -> str`

Process security labels with proper error handling.

### `clean_text(text: str) -> str`

Clean and validate text content.

### `extract_field_value(field: Dict) -> Any`

Extract value from a field dictionary.

### `extract_custom_fields(fields: Dict) -> Dict[str, Any]`

Extract all custom fields that have non-null values.

### `extract_technical_info(text: str) -> Dict[str, Any]`

Extract technical information from ticket text.

### `clean_html_content(text: str) -> str`

Remove HTML and standardize format with better error handling.

### `extract_ticket_metadata(record: Dict) -> Dict`

Extract relevant metadata for DevOps context.

### `categorize_tech_stack(text: str, components: List[str]) -> Dict[str, bool]`

Categorize ticket into technical domains.

### `calculate_impact_score(record: Dict) -> float`

Calculate ticket impact score based on multiple factors.

### `extract_sla_metrics(record: Dict) -> Dict`

Extract SLA and response time metrics with proper error handling.

### `clean_data(raw_tickets: list) -> pd.DataFrame`

Enhanced cleaning pipeline for DevOps tickets.

## Example Code

Here is an example of the `data_preprocessing.py` script:

```python
import pandas as pd
import numpy as np  # Add numpy import
import re
from datetime import datetime
from dateutil.parser import parse
from typing import Any, Optional, Union, List, Dict
import logging
import sys
from ai_ticket_processing import analyze_ticket_severity, extract_incident_patterns, get_analyzer
from bs4 import BeautifulSoup
from bs4 import MarkupResemblesLocatorWarning, XMLParsedAsHTMLWarning
import warnings

# Suppress XML parsing warning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Filter BeautifulSoup warnings
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('processing.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

def safe_get(data: Any, path: str, default: Any = None) -> Any:
    """Robust nested data access with type checking"""
    if not isinstance(data, dict):
        return default
        
    keys = path.split('.')
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current if current is not None else default

def process_datetime(value: Any) -> Optional[datetime]:
    """Safe datetime parsing with correct type hints"""
    try:
        if pd.isna(value):
            return pd.NaT
        return parse(str(value))
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid datetime value: {value} - {str(e)}")
        return pd.NaT

def process_components(components: List[Dict]) -> str:
    """Process components with proper error handling"""
    if not isinstance(components, list):
        return ""
    try:
        return ", ".join([
            str(c.get("name", "")) for c in components 
            if isinstance(c, dict) and isinstance(c.get("name"), str)
        ])
    except Exception as e:
        logger.warning(f"Error processing components: {str(e)}")
        return ""

def process_labels(labels: List[str]) -> str:
    """Process security labels with proper error handling"""
    if not isinstance(labels, list):
        return ""
    try:
        return ", ".join([
            lbl for lbl in labels 
            if isinstance(lbl, str) and re.search(r"sec|vuln", lbl, re.I)
        ])
    except Exception as e:
        logger.warning(f"Error processing labels: {str(e)}")
        return ""

def clean_text(text: str) -> str:
    """Clean and validate text content"""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if len(text) < 10:  # Skip very short texts
        return ""
    return text

def extract_field_value(field: Dict) -> Any:
    """Extract value from a field dictionary"""
    if not isinstance(field, dict):
        return field
        
    # Handle common field patterns
    if "value" in field:
        return field["value"]
    if "name" in field:
        return field["name"]
    if "displayName" in field:
        return field["displayName"]
        
    return field

def extract_custom_fields(fields: Dict) -> Dict[str, Any]:
    """Extract all custom fields that have non-null values"""
    custom_fields = {}
    
    for key, value in fields.items():
        if key.startswith("customfield_") and value is not None:
            cleaned_value = extract_field_value(value)
            if cleaned_value not in [None, "", "9223372036854775807"]:  # Skip default/empty values
                custom_fields[key] = cleaned_value
                
    return custom_fields

def extract_technical_info(text: str) -> Dict[str, Any]:
    """Extract technical information from ticket text"""
    tech_patterns = {
        'has_error_msg': bool(re.search(r'error|exception|failed|crash', text, re.I)),
        'has_stacktrace': bool(re.search(r'at .+\(.+\)|exception in thread', text, re.I)),
        'has_version': bool(re.search(r'version|v\d+\.\d+|release', text, re.I)),
        'has_path': bool(re.search(r'[/\\](?:[a-zA-Z0-9-_. ]+[/\\])+[a-zA-Z0-9-_.]+', text)),
        'has_config': bool(re.search(r'config|setting|parameter|env|variable', text, re.I))
    }
    return tech_patterns

def clean_html_content(text: str) -> str:
    """Remove HTML and standardize format with better error handling"""
    if not isinstance(text, str):
        return ""
        
    try:
        # Pre-process text to handle common issues
        # Remove common file paths and URLs that trigger warnings
        text = re.sub(r'(?:file|https?|ftp)://[^\s<>"]+|www\.[^\s<>"]+', ' ', text)
        text = re.sub(r'[/\\][\w\-. /\\]+\.\w{2,4}', ' ', text)
        
        # Handle HTML/XML content
        if re.search(r'<[^>]+>', text):
            text = BeautifulSoup(text, 'lxml', parser_type='xml').get_text()
        
        # Standardize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
        
    except Exception as e:
        logger.debug(f"HTML cleaning failed: {str(e)}")
        # Fallback to basic cleaning
        return re.sub(r'\s+', ' ', text).strip()

def extract_ticket_metadata(record: Dict) -> Dict:
    """Extract relevant metadata for DevOps context"""
    metadata = {
        'time_to_resolve': None,
        'component_count': 0,
        'has_attachments': False,
        'comment_count': 0,
        'technical_priority': 0
    }
    
    # Calculate resolution time
    if record.get('created') and record.get('updated'):
        try:
            created = pd.to_datetime(record['created'])
            updated = pd.to_datetime(record['updated'])
            metadata['time_to_resolve'] = (updated - created).total_seconds()
        except:
            pass
    
    # Count components
    if isinstance(record.get('components'), list):
        metadata['component_count'] = len(record['components'])
    
    # Check for technical indicators
    desc = str(record.get('description', ''))
    tech_info = extract_technical_info(desc)
    metadata['technical_priority'] = sum(tech_info.values())
    
    # Count comments
    if isinstance(record.get('comments'), list):
        metadata['comment_count'] = len(record['comments'])
    
    return metadata

def categorize_tech_stack(text: str, components: List[str]) -> Dict[str, bool]:
    """Categorize ticket into technical domains"""
    categories = {
        'infrastructure': r'server|cloud|kubernetes|docker|aws|azure|gcp|infra|vm|vpc|network',
        'database': r'sql|mongodb|database|db|redis|postgresql|mysql|nosql',
        'security': r'security|vulnerability|auth|csrf|xss|breach|password|encryption',
        'application': r'api|frontend|backend|app|service|endpoint|ui|interface',
        'deployment': r'deploy|ci/cd|pipeline|release|build|jenkins|git',
        'monitoring': r'monitor|alert|log|metric|grafana|prometheus|trace'
    }
    
    results = {}
    combined_text = f"{text} {' '.join(components)}".lower()
    
    for category, pattern in categories.items():
        results[f"is_{category}"] = bool(re.search(pattern, combined_text, re.I))
    
    return results

def calculate_impact_score(record: Dict) -> float:
    """Calculate ticket impact score based on multiple factors"""
    score = 0.0
    
    # Technical complexity
    score += record['technical_priority'] * 0.3
    
    # Service impact
    if record.get('is_infrastructure', False):
        score += 2.0
    
    # Time sensitivity
    if record.get('time_to_resolve'):
        if record['time_to_resolve'] < 3600:  # 1 hour
            score += 2.0
        elif record['time_to_resolve'] < 86400:  # 24 hours
            score += 1.0
    
    # Comment activity indicates importance
    comment_count = record.get('comment_count', 0)
    score += min(comment_count * 0.1, 1.0)
    
    # Normalize to 0-10 scale
    return min(score, 10.0)

def extract_sla_metrics(record: Dict) -> Dict:
    """Extract SLA and response time metrics with proper error handling"""
    metrics = {
        'response_time_mins': None,
        'resolution_time_mins': None,
        'sla_breached': False,
        'high_urgency': False
    }
    
    try:
        # Convert priority to string if it's a dict
        priority = record.get('priority', 'Medium')
        if isinstance(priority, dict):
            priority = priority.get('name', 'Medium')
            
        # Handle timestamps
        created = record.get('created')
        updated = record.get('updated')
        
        if created and updated:
            created = pd.to_datetime(created) if isinstance(created, str) else created
            updated = pd.to_datetime(updated) if isinstance(updated, str) else updated
            
            if pd.notnull(created) and pd.notnull(updated):
                resolution_time = (updated - created).total_seconds() / 60
                metrics['resolution_time_mins'] = resolution_time
                
                # SLA thresholds in minutes
                sla_thresholds = {
                    'Critical': 60,    # 1 hour
                    'High': 240,      # 4 hours
                    'Medium': 1440,   # 24 hours
                    'Low': 4320       # 72 hours
                }
                
                threshold = sla_thresholds.get(str(priority), 1440)
                metrics['sla_breached'] = resolution_time > threshold
                metrics['high_urgency'] = resolution_time < threshold / 2
                
    except Exception as e:
        logger.debug(f"SLA calculation failed: {str(e)} - Priority: {priority}")
        
    return metrics

def clean_data(raw_tickets: list) -> pd.DataFrame:
    """Enhanced cleaning pipeline for DevOps tickets"""
    if not isinstance(raw_tickets, list):
        logger.error("Input must be a list of tickets")
        return pd.DataFrame()
    
    processed = []
    for idx, ticket in enumerate(raw_tickets):
        try:
            fields = safe_get(ticket, "fields", {})
            
            # Basic record with enhanced cleaning
            record = {
                "_id": str(safe_get(ticket, "_id.$oid", f"missing_id_{idx}")),
                "key": safe_get(ticket, "key", ""),
                "created": process_datetime(safe_get(fields, "created")),
                "updated": process_datetime(safe_get(fields, "updated")),
                "summary": clean_html_content(safe_get(fields, "summary", "")),
                "description": clean_html_content(safe_get(fields, "description", "")),
                "priority": extract_field_value(safe_get(fields, "priority", {})),
                "status": extract_field_value(safe_get(fields, "status", {})),
                "issue_type": extract_field_value(safe_get(fields, "issuetype", {})),
            }
            
            # Extract technical context
            components = safe_get(fields, "components", [])
            tech_categories = categorize_tech_stack(
                record["description"], 
                [c.get("name", "") for c in components]
            )
            record.update(tech_categories)
            
            # Add metadata and metrics
            metadata = extract_ticket_metadata(record)
            sla_metrics = extract_sla_metrics(record)
            if any(sla_metrics.values()):  # Only update if we got valid metrics
                record.update(sla_metrics)
            record.update(metadata)
            
            # Calculate impact and priority scores
            record["impact_score"] = calculate_impact_score(record)
            record["tech_score"] = min(10, metadata["technical_priority"] * 2)
            
            processed.append(record)
            
        except Exception as e:
            logger.exception(f"Failed to process ticket {idx}")
            continue
    
    df = pd.DataFrame(processed)
    
    if not df.empty:
        # Ensure consistent columns
        required_columns = [
            '_id', 'key', 'created', 'updated', 'summary', 'description',
            'priority', 'status', 'issue_type', 'components', 'technical_priority',
            'impact_score', 'tech_score'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
        
        # Remove any unexpected columns
        df = df[required_columns + [col for col in df.columns if col not in required_columns]]
        
        # Calculate final priority score (1-5 scale)
        df["calculated_priority"] = df.apply(
            lambda x: min(5, max(1, round(
                (x["impact_score"] / 2) + 
                (x["tech_score"] / 4) +
                (2 if x["sla_breached"] else 0) +
                (1 if x["high_urgency"] else 0)
            ))), axis=1
        )
        
        # Add ticket category based on technical aspects
        df["ticket_category"] = df.apply(
            lambda x: "Infrastructure" if x["is_infrastructure"]
            else "Security" if x["is_security"]
            else "Application" if x["is_application"]
            else "Deployment" if x["is_deployment"]
            else "Database" if x["is_database"]
            else "Monitoring" if x["is_monitoring"]
            else "Other", axis=1
        )
        
        # Convert boolean columns to integers
        bool_cols = ['sla_breached', 'high_urgency', 'has_attachments']
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].map({
                    True: 1, 'True': 1, 'true': 1,
                    False: 0, 'False': 0, 'false': 0,
                    None: 0, 'None': 0, np.nan: 0
                }).fillna(0).astype('int32')
        
        # Convert technical priority to float
        if 'technical_priority' in df.columns:
            df['technical_priority'] = pd.to_numeric(
                df['technical_priority'].replace(
                    {'True': '1', 'False': '0', True: '1', False: '0'}
                ),
                errors='coerce'
            ).fillna(0).astype('float32')
        
        # Ensure numeric types for key metrics
        numeric_cols = {
            'impact_score': 'float32',
            'tech_score': 'float32',
            'comment_count': 'int32',
            'component_count': 'int32'
        }
        
        for col, dtype in numeric_cols.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(dtype)
    
    logger.info(f"Processed {len(df)} tickets with enhanced IT operations context")
    return df
```

## Notes

- Ensure that the necessary dependencies are installed before executing the data preprocessing script.
- The `processing.log` file is used to log the preprocessing steps and any errors encountered during the process.
