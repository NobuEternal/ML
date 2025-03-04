# Configuration Settings

## Purpose
The configuration settings in this project are used to define various parameters and options that control the behavior of the system. These settings allow for customization and flexibility in how the system operates.

## Configuration Settings

### 1. MongoDB Configuration
- **MONGODB_URI**: The URI for connecting to the MongoDB database.
- **DATABASE_NAME**: The name of the database to use.
- **COLLECTION_NAME**: The name of the collection within the database.

#### Example Configuration
```python
# config.py
MONGODB_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "JiraRepos"
COLLECTION_NAME = "Jira"
```

### 2. Model Training Configuration
- **min_accuracy_threshold**: The minimum accuracy threshold below which retraining is triggered.
- **retrain_threshold**: The minimum number of new samples required before retraining.

#### Example Configuration
```python
# auto_retrain.py
class AutoRetrainer:
    def __init__(self, monitor: ModelMonitor):
        self.monitor = monitor
        self.min_accuracy_threshold = 0.85
        self.retrain_threshold = 1000
```

### 3. Field Configuration
- **priority_mapping**: Mapping of priority levels to numerical values.
- **security_indicators**: Custom fields and labels used to identify security-related tickets.
- **devops_components**: List of components related to DevOps.

#### Example Configuration
```yaml
# field_config.yaml
priority_mapping:
  field: fields.priority.name
  values:
    Critical: 0
    High: 1
    Medium: 2
    Low: 3

security_indicators:
  custom_fields:
    - fields.customfield_12345  # Attack Vector
    - fields.customfield_67890  # CVSS Score
  labels:
    - "security"
    - "pentest"
    - "vulnerability"

devops_components:
  - "kubernetes"
  - "aws"
  - "azure"
  - "pipeline"
  - "deployment"
```

## Modifying Configuration Settings
To modify the configuration settings, follow these steps:

1. **Open the configuration file**: Locate the configuration file you want to modify (e.g., `config.py`, `field_config.yaml`).
2. **Edit the settings**: Update the values of the settings as needed.
3. **Save the file**: Save the changes to the configuration file.

By following these instructions, you can customize the configuration settings to suit your specific requirements.
