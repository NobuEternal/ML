# CUDA Test Script

## Purpose
The CUDA test script is used to verify the availability and functionality of CUDA on the system. It checks if CUDA is available, the version of CUDA, and the GPU device name.

## Functionality
The script performs the following checks:
1. Checks if CUDA is available on the system.
2. Prints the PyTorch version.
3. Prints the CUDA version if available.
4. Prints the GPU device name if available.

## Running the CUDA Test Script
To run the CUDA test script, follow these steps:

1. Ensure you have PyTorch installed. You can install it using the following command:
   ```bash
   pip install torch
   ```

2. Run the CUDA test script:
   ```bash
   python cuda_test.py
   ```

## Interpreting the Results
The script will output the following information:
- PyTorch version
- Whether CUDA is available
- CUDA version (if available)
- GPU device name (if available)

Example output:
```
=== CUDA Diagnostic ===
PyTorch version: 1.9.0
CUDA available: True
CUDA version: 11.1
GPU Device: NVIDIA GeForce GTX 1080 Ti
```

If CUDA is not available, the output will indicate that CUDA is not detected.
