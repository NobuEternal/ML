import pandas as pd
import re
from datetime import datetime
from dateutil.parser import parse
from typing import Any, Optional, Union, List, Dict
import logging
import sys
from ai_ticket_processing import analyze_ticket_severity, extract_incident_patterns

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

def clean_data(raw_tickets: list) -> pd.DataFrame:
    """Main data cleaning pipeline with full type safety"""
    if not isinstance(raw_tickets, list):
        logger.error("Input must be a list of tickets")
        return pd.DataFrame()
        
    processed = []
    
    for idx, ticket in enumerate(raw_tickets):
        try:
            # Validate base document structure
            if not isinstance(ticket, dict):
                logger.error(f"Skipping non-dict ticket at index {idx}")
                continue
                
            # Extract core fields
            fields = safe_get(ticket, "fields", {})
            if not isinstance(fields, dict):
                logger.error(f"Invalid fields type in ticket {safe_get(ticket, '_id.$oid', 'unknown')}")
                continue

            # Create base record
            record = {
                "_id": str(safe_get(ticket, "_id.$oid", f"missing_id_{idx}")),
                "key": safe_get(ticket, "key", ""),
                "created": process_datetime(safe_get(fields, "created")),
                "updated": process_datetime(safe_get(fields, "updated")),
                "summary": safe_get(fields, "summary", ""),
                "description": safe_get(fields, "description", ""),
                "priority": safe_get(fields, "priority.name", "Medium"),
                "status": safe_get(fields, "status.name", "Open"),
                "issue_type": safe_get(fields, "issuetype.name", "Task")
            }

            # Process components and labels
            record["components"] = process_components(safe_get(fields, "components", []))
            record["security_labels"] = process_labels(safe_get(fields, "labels", []))

            # Add AI analysis with error handling
            try:
                record["severity"] = analyze_ticket_severity(record["description"])
                incident_data = extract_incident_patterns(record["description"])
                if isinstance(incident_data, dict):
                    record.update(incident_data)
                else:
                    logger.warning(f"Invalid incident data for ticket {record['_id']}")
                    record["incident_type"] = "unknown"
            except Exception as e:
                logger.warning(f"AI analysis failed for ticket {record['_id']}: {str(e)}")
                record["severity"] = 0.0
                record["incident_type"] = "unknown"

            processed.append(record)
            logger.info(f"Processed ticket {record['_id']} with key {record['key']}")
            
        except Exception as e:
            logger.exception(f"Failed to process ticket index {idx}")
            
    # Create DataFrame and handle empty case
    if not processed:
        logger.warning("No tickets were successfully processed")
        return pd.DataFrame(columns=[
            "_id", "key", "created", "updated", "summary", "description",
            "priority", "status", "components", "security_labels", "severity",
            "incident_type", "is_devops", "is_security"
        ])
    
    df = pd.DataFrame(processed)
    
    # Add boolean flags with error handling
    try:
        df["is_devops"] = df["components"].str.contains(
            r"server|cloud|kubernetes|aws", case=False, na=False
        ).astype(int)
        
        df["is_security"] = (
            (df["severity"] > 0.7) |
            df["security_labels"].str.contains("sec|vuln", na=False)
        ).astype(int)
    except Exception as e:
        logger.error(f"Error creating boolean flags: {str(e)}")
        df["is_devops"] = 0
        df["is_security"] = 0
    
    # Ensure all required columns are present
    required_columns = [
        "_id", "key", "created", "updated", "summary", "description",
        "priority", "status", "components", "security_labels", "severity",
        "incident_type", "is_devops", "is_security"
    ]
    
    for col in required_columns:
        if col not in df.columns:
            df[col] = None
            
    logger.info(f"Cleaned data frame with {len(df)} records")
    return df[required_columns]
