# Field Configuration Settings

## Purpose
The field configuration settings in this project are used to define various parameters and options that control the behavior of the system. These settings allow for customization and flexibility in how the system operates.

## Field Configuration Settings

### 1. Priority Mapping
- **priority_mapping**: Mapping of priority levels to numerical values.

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
```

### 2. Security Indicators
- **security_indicators**: Custom fields and labels used to identify security-related tickets.

#### Example Configuration
```yaml
# field_config.yaml
security_indicators:
  custom_fields:
    - fields.customfield_12345  # Attack Vector
    - fields.customfield_67890  # CVSS Score
  labels:
    - "security"
    - "pentest"
    - "vulnerability"
```

### 3. DevOps Components
- **devops_components**: List of components related to DevOps.

#### Example Configuration
```yaml
# field_config.yaml
devops_components:
  - "kubernetes"
  - "aws"
  - "azure"
  - "pipeline"
  - "deployment"
```

## Modifying Field Configuration Settings
To modify the field configuration settings, follow these steps:

1. **Open the configuration file**: Locate the `field_config.yaml` file.
2. **Edit the settings**: Update the values of the settings as needed.
3. **Save the file**: Save the changes to the configuration file.

By following these instructions, you can customize the field configuration settings to suit your specific requirements.
