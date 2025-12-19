# Examples Directory

This directory contains example scripts demonstrating how to use the semantic-decoding codebase.

## Available Examples

### load_data_example.py

Demonstrates how to load fMRI data with corresponding text stimuli.

**What it shows:**
- Loading fMRI response data for specific subjects and stories
- Loading text stimulus data with word timing information
- Verifying temporal alignment between fMRI and text
- Different data access patterns (stacked, dictionary, subsets)

**Usage:**
```bash
# Make sure you have downloaded the training data first
python examples/load_data_example.py
```

**Prerequisites:**
- Training data must be downloaded (see main README.md)
- Data should be in `data_train/` directory

**Expected output:**
The script will display:
- Shape and statistics of loaded fMRI data
- Word sequences with timing information
- Data alignment verification
- Examples of different access patterns

## See Also

- [LOADING_DATA_TUTORIAL.md](../LOADING_DATA_TUTORIAL.md) - Comprehensive guide
- [QUICK_REFERENCE.md](../QUICK_REFERENCE.md) - Quick reference
- Main [README.md](../README.md) - Repository overview
