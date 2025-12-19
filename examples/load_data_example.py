#!/usr/bin/env python3
"""
Example script demonstrating how to load fMRI data with corresponding text stimuli.

This script shows the basic workflow for loading and inspecting fMRI responses
and text stimuli from the semantic-decoding dataset.

Usage:
    python examples/load_data_example.py

Prerequisites:
    - Download training data as described in README.md
    - Ensure data_train/ directory is properly set up
"""

import os
import sys
import numpy as np

# Add parent directory to path to import from decoding module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from decoding.utils_resp import get_resp
from decoding.utils_stim import get_story_wordseqs
from decoding import config


def main():
    """Main example function demonstrating data loading."""
    
    print("=" * 70)
    print("Example: Loading fMRI Data with Corresponding Text")
    print("=" * 70)
    
    # Configuration
    # Note: Change these values based on your downloaded data
    subject = "UTS03"
    stories = ["alternateithicatom"]  # Use stories available in your data
    
    print(f"\nSubject: {subject}")
    print(f"Stories: {stories}")
    
    # -------------------------------------------------------------------------
    # Step 1: Load fMRI Response Data
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Step 1: Loading fMRI Response Data")
    print("-" * 70)
    
    try:
        # Load responses for specified stories
        # Returns array of shape [total_TRs, n_voxels]
        resp = get_resp(subject, stories, stack=True)
        
        print(f"✓ Successfully loaded fMRI data")
        print(f"  Shape: {resp.shape}")
        print(f"  - Number of TRs (time points): {resp.shape[0]}")
        print(f"  - Number of voxels: {resp.shape[1]}")
        print(f"  - Data range: [{resp.min():.2f}, {resp.max():.2f}]")
        print(f"  - Mean activation: {resp.mean():.2f}")
        
    except FileNotFoundError as e:
        print(f"✗ Error: Could not find fMRI data file")
        print(f"  Make sure you have downloaded the training data")
        print(f"  Expected location: {config.DATA_TRAIN_DIR}/train_response/{subject}/")
        return
    
    # -------------------------------------------------------------------------
    # Step 2: Load Text Stimulus Data
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Step 2: Loading Text Stimulus Data")
    print("-" * 70)
    
    try:
        # Load word sequences with timing information
        # Returns dict: {story_name: DataSequence object}
        wordseqs = get_story_wordseqs(stories)
        
        for story in stories:
            ds = wordseqs[story]
            print(f"✓ Successfully loaded text for story: {story}")
            print(f"  - Number of words: {len(ds.data)}")
            print(f"  - Number of TRs: {len(ds.tr_times)}")
            print(f"  - Duration: {ds.data_times[-1]:.2f} seconds")
            
            # Show first few words with timing
            print(f"\n  First 10 words with timing:")
            for i in range(min(10, len(ds.data))):
                print(f"    {i+1:2d}. '{ds.data[i]:15s}' at {ds.data_times[i]:6.2f}s")
            
            # Show TR timing
            print(f"\n  TR acquisition times (first 5):")
            for i in range(min(5, len(ds.tr_times))):
                print(f"    TR {i+1:2d}: {ds.tr_times[i]:6.2f}s")
                
    except FileNotFoundError as e:
        print(f"✗ Error: Could not find text stimulus files")
        print(f"  Make sure you have downloaded the training data")
        print(f"  Expected location: {config.DATA_TRAIN_DIR}/train_stimulus/")
        return
    
    # -------------------------------------------------------------------------
    # Step 3: Verify Data Alignment
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Step 3: Verifying Data Alignment")
    print("-" * 70)
    
    for story in stories:
        ds = wordseqs[story]
        expected_trs = len(ds.tr_times)
        actual_trs = resp.shape[0]
        
        print(f"\nStory: {story}")
        print(f"  fMRI data has {actual_trs} TRs")
        print(f"  Text data expects {expected_trs} TRs")
        
        # Note: There might be trimming applied during feature extraction
        # The stimulus processing trims TRs (config.TRIM = 5)
        if actual_trs >= expected_trs - 2 * config.TRIM:
            print(f"  ✓ Alignment looks good (accounting for trimming)")
        else:
            print(f"  ⚠ Warning: TR count mismatch may cause issues")
    
    # -------------------------------------------------------------------------
    # Step 4: Example Data Access Patterns
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Step 4: Example Data Access Patterns")
    print("-" * 70)
    
    # Load data without stacking (as dictionary)
    resp_dict = get_resp(subject, stories, stack=False)
    print(f"\n✓ Loaded responses as dictionary:")
    for story in stories:
        print(f"  - {story}: shape {resp_dict[story].shape}")
    
    # Load only specific voxels
    n_voxels_subset = 100
    voxels_to_load = np.arange(n_voxels_subset)
    resp_subset = get_resp(subject, stories, stack=True, vox=voxels_to_load)
    print(f"\n✓ Loaded subset of voxels:")
    print(f"  - Requested: {len(voxels_to_load)} voxels")
    print(f"  - Result shape: {resp_subset.shape}")
    
    # Access DataSequence attributes
    print(f"\n✓ DataSequence object attributes:")
    ds = wordseqs[stories[0]]
    print(f"  - ds.data: List of {len(ds.data)} words")
    print(f"  - ds.data_times: Array of shape {ds.data_times.shape}")
    print(f"  - ds.tr_times: Array of shape {ds.tr_times.shape}")
    print(f"  - ds.split_inds: Array of shape {ds.split_inds.shape}")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"""
This example demonstrated:
1. Loading fMRI response data using get_resp()
2. Loading text stimulus data using get_story_wordseqs()
3. Verifying temporal alignment between fMRI and text
4. Accessing data in different formats (stacked, dictionary, subsets)

Key Points:
- fMRI data: {resp.shape[0]} TRs × {resp.shape[1]} voxels
- Text data: {len(wordseqs[stories[0]].data)} words over {ds.data_times[-1]:.1f} seconds
- TR sampling rate: ~{np.mean(np.diff(ds.tr_times)):.2f} seconds

For more details, see:
- LOADING_DATA_TUTORIAL.md (comprehensive guide)
- QUICK_REFERENCE.md (quick lookup)
    """)
    
    print("=" * 70)


if __name__ == "__main__":
    main()
