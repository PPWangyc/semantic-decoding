# Tutorial: How to Load fMRI Data with Corresponding Text

This tutorial explains step-by-step how to load fMRI brain recordings (responses) along with their corresponding text stimuli in the semantic-decoding repository.

## Overview

The repository processes two main types of data:
1. **fMRI Response Data**: Brain activity recordings stored as `.hf5` (HDF5) files
2. **Text Stimulus Data**: Story transcripts with word timing information stored as `.TextGrid` files

## Data Directory Structure

```
semantic-decoding/
├── data_train/
│   ├── train_response/
│   │   └── [SUBJECT_ID]/
│   │       └── [STORY_NAME].hf5        # fMRI data files
│   ├── train_stimulus/
│   │   └── [STORY_NAME].TextGrid       # Text with timing
│   ├── respdict.json                   # Response metadata
│   └── sess_to_story.json              # Session to story mapping
├── data_test/
│   ├── test_response/
│   │   └── [SUBJECT_ID]/
│   │       └── [EXPERIMENT]/
│   │           └── [TASK].hf5          # Test fMRI data
│   └── test_stimulus/
│       └── [EXPERIMENT]/               # Test stimuli
└── data_lm/                            # Language model data
```

## Step-by-Step Guide

### Step 1: Understanding the Data Formats

#### fMRI Response Data (.hf5 files)
- Format: HDF5 files containing brain activity measurements
- Shape: `[n_TRs, n_voxels]` where:
  - `n_TRs`: Number of time repetitions (brain scans)
  - `n_voxels`: Number of brain voxels (spatial measurements)
- Each TR (Time Repetition) is typically acquired every ~2 seconds

#### Text Stimulus Data (.TextGrid files)
- Format: Praat TextGrid files containing word transcripts with precise timing
- Contains: Words and their start/end times aligned with the audio stimulus
- Organized in tiers (phonemes, words, characters, etc.)

### Step 2: Loading fMRI Response Data

Use the `get_resp()` function from `utils_resp.py`:

```python
import numpy as np
from decoding.utils_resp import get_resp

# Load fMRI responses for specific stories
subject = "UTS03"  # Subject ID
stories = ["alternateithicatom", "wheretheressmoke"]  # Story names

# Load and stack all stories into a single matrix
resp = get_resp(subject, stories, stack=True)
# Returns: numpy array of shape [total_TRs, n_voxels]

# Or load as a dictionary (one entry per story)
resp_dict = get_resp(subject, stories, stack=False)
# Returns: dict with story names as keys, each containing [n_TRs, n_voxels]

# Load specific voxels only
voxels_to_load = np.array([0, 1, 2, 100, 200])  # Voxel indices
resp_subset = get_resp(subject, stories, stack=True, vox=voxels_to_load)
# Returns: numpy array of shape [total_TRs, len(voxels_to_load)]
```

**Implementation details** (`utils_resp.py`):
```python
def get_resp(subject, stories, stack=True, vox=None):
    """Loads response data
    
    Args:
        subject: Subject identifier (e.g., "UTS03")
        stories: List of story names
        stack: If True, vertically stack all stories; if False, return dict
        vox: Optional array of voxel indices to load (loads all if None)
    
    Returns:
        If stack=True: numpy array [total_TRs, n_voxels]
        If stack=False: dict {story_name: array [n_TRs, n_voxels]}
    """
```

### Step 3: Loading Text Stimulus Data

Use the `get_story_wordseqs()` function from `utils_stim.py`:

```python
import json
from decoding.utils_stim import get_story_wordseqs
import config

# Load word sequences with timing for stories
stories = ["alternateithicatom", "wheretheressmoke"]
wordseqs = get_story_wordseqs(stories)

# wordseqs is a dict: {story_name: DataSequence object}
for story in stories:
    ds = wordseqs[story]
    
    # Access the words
    words = ds.data  # List of words in the story
    
    # Access timing information
    word_times = ds.data_times  # Average time of each word
    tr_times = ds.tr_times      # TR acquisition times
    
    # Print first 10 words with their times
    for i in range(min(10, len(words))):
        print(f"Word: {words[i]}, Time: {word_times[i]:.2f}s")
```

**DataSequence Object Structure**:
- `data`: List of words (strings)
- `data_times`: Array of word timing (center time of each word)
- `tr_times`: Array of TR acquisition times
- `split_inds`: Indices where data splits into TR chunks

### Step 4: Complete Example - Loading Both fMRI and Text Together

Here's a complete example showing how to load matched fMRI data and text stimuli:

```python
import os
import json
import numpy as np
from decoding import config
from decoding.utils_resp import get_resp
from decoding.utils_stim import get_story_wordseqs

# 1. Define subject and stories
subject = "UTS03"
stories = ["alternateithicatom"]

# 2. Load fMRI responses
print("Loading fMRI responses...")
resp = get_resp(subject, stories, stack=True)
print(f"Response shape: {resp.shape}")  # [n_TRs, n_voxels]

# 3. Load text with timing
print("\nLoading text stimuli...")
wordseqs = get_story_wordseqs(stories)

for story in stories:
    ds = wordseqs[story]
    print(f"\nStory: {story}")
    print(f"Number of words: {len(ds.data)}")
    print(f"Number of TRs: {len(ds.tr_times)}")
    print(f"Story duration: {ds.data_times[-1]:.2f} seconds")
    
    # Display first few words
    print("\nFirst 5 words with timing:")
    for i in range(min(5, len(ds.data))):
        print(f"  {i+1}. '{ds.data[i]}' at {ds.data_times[i]:.2f}s")
    
    # Show TR times
    print(f"\nFirst 3 TR times: {ds.tr_times[:3]}")

# 4. Verify alignment
print(f"\nAlignment check:")
print(f"Response has {resp.shape[0]} TRs")
print(f"Stimulus expects {len(wordseqs[stories[0]].tr_times)} TRs")
```

### Step 5: Extract Features from Text for Encoding Models

To use the text data for encoding models, you need to extract features:

```python
from decoding.GPT import GPT
from decoding.StimulusModel import LMFeatures
from decoding.utils_stim import get_stim
import config

# 1. Load GPT model for feature extraction
with open(os.path.join(config.DATA_LM_DIR, "perceived", "vocab.json"), "r") as f:
    gpt_vocab = json.load(f)

gpt = GPT(
    path=os.path.join(config.DATA_LM_DIR, "perceived", "model"),
    vocab=gpt_vocab,
    device=config.GPT_DEVICE
)

# 2. Create feature extractor
features = LMFeatures(
    model=gpt,
    layer=config.GPT_LAYER,      # Which GPT layer to extract from (default: 9)
    context_words=config.GPT_WORDS  # Number of context words (default: 5)
)

# 3. Extract stimulus features aligned to TRs
stories = ["alternateithicatom"]
stim, tr_stats, word_stats = get_stim(stories, features)

print(f"Stimulus features shape: {stim.shape}")
# Returns: [n_TRs, n_features] where features are delayed/convolved
```

### Step 6: Working with Test Data

Test data follows a similar structure but with different organization:

```python
import h5py
import os
from decoding import config

# Test data parameters
subject = "UTS03"
experiment = "perceived_speech"  # or "imagined_speech", "perceived_movies"
task = "wheretheressmoke"

# Load test response data
test_resp_path = os.path.join(
    config.DATA_TEST_DIR,
    "test_response",
    subject,
    experiment,
    task + ".hf5"
)

with h5py.File(test_resp_path, "r") as hf:
    test_resp = np.nan_to_num(hf["data"][:])
    print(f"Test response shape: {test_resp.shape}")
```

## Key Functions Reference

### `get_resp(subject, stories, stack=True, vox=None)`
**Location**: `decoding/utils_resp.py`
- **Purpose**: Load fMRI response data
- **Input**: Subject ID, list of story names
- **Output**: Brain activity matrix [TRs × voxels]

### `get_story_wordseqs(stories)`
**Location**: `decoding/utils_stim.py`
- **Purpose**: Load word sequences with timing
- **Input**: List of story names
- **Output**: Dict of DataSequence objects with words and timing

### `get_stim(stories, features, tr_stats=None)`
**Location**: `decoding/utils_stim.py`
- **Purpose**: Extract quantitative features from text stimuli
- **Input**: Story names and feature extractor
- **Output**: Feature matrix aligned to TR times [TRs × features]

## Understanding Data Alignment

The key to this system is proper temporal alignment:

1. **Words** are presented at specific times during the story
2. **fMRI TRs** are acquired every ~2 seconds
3. **Lanczos interpolation** is used to downsample word-level features to TR-level features
4. **Hemodynamic delays** (config.STIM_DELAYS = [1,2,3,4]) account for the brain's delayed response

```
Timeline:
---------
Words:     w1   w2   w3   w4   w5   w6   w7   w8   ...
           |    |    |    |    |    |    |    |
Time:      0.2  0.8  1.3  2.1  2.7  3.5  4.2  4.9  ... (seconds)
           
TRs:       |         TR1        |         TR2        |
           0s                   2s                   4s
           
Features are interpolated from word times to TR times with delays
```

## Common Issues and Solutions

### Issue 1: File Not Found
**Problem**: `FileNotFoundError: [STORY_NAME].hf5 not found`
**Solution**: Download the data from the sources listed in README.md:
- Training data: https://utexas.box.com/shared/static/3go1g4gcdar2cntjit2knz5jwr3mvxwe.zip
- Test data: https://utexas.box.com/shared/static/ae5u0t3sh4f46nvmrd3skniq0kk2t5uh.zip

### Issue 2: Shape Mismatch
**Problem**: Response and stimulus have different numbers of TRs
**Solution**: The stimulus processing trims TRs (config.TRIM = 5). Account for this:
```python
# Stimulus is trimmed: [5+TRIM : -TRIM]
# So if you have 100 TRs in response, stimulus will have 90 TRs
```

### Issue 3: Memory Issues
**Problem**: Loading all voxels uses too much memory
**Solution**: Load only selected voxels:
```python
# Load only top 10,000 voxels (as in encoding model)
voxels = np.load("path/to/encoding_model.npz")["voxels"]
resp = get_resp(subject, stories, stack=True, vox=voxels)
```

## Advanced: Complete Pipeline Example

Here's how the training script uses these functions together:

```python
# From train_EM.py - simplified
import numpy as np
import json
from decoding import config
from decoding.utils_resp import get_resp
from decoding.utils_stim import get_stim
from decoding.GPT import GPT
from decoding.StimulusModel import LMFeatures

# 1. Get training stories from session mapping
with open(os.path.join(config.DATA_TRAIN_DIR, "sess_to_story.json"), "r") as f:
    sess_to_story = json.load(f)
stories = []
for sess in [2, 3, 4, 5]:  # Example sessions
    stories.extend(sess_to_story[str(sess)])

# 2. Load GPT for feature extraction
with open(os.path.join(config.DATA_LM_DIR, "perceived", "vocab.json"), "r") as f:
    gpt_vocab = json.load(f)
gpt = GPT(os.path.join(config.DATA_LM_DIR, "perceived", "model"), gpt_vocab)
features = LMFeatures(gpt, layer=9, context_words=5)

# 3. Load and align data
rstim, tr_stats, word_stats = get_stim(stories, features)
rresp = get_resp("UTS03", stories, stack=True)

print(f"Stimulus shape: {rstim.shape}")   # [n_TRs, n_features * n_delays]
print(f"Response shape: {rresp.shape}")   # [n_TRs, n_voxels]

# 4. Now rstim and rresp are aligned and ready for encoding model training!
# Each row corresponds to the same TR (time point)
```

## Summary

To load fMRI data with corresponding text:

1. **Install and Download**: Set up the repository and download data files
2. **Load fMRI**: Use `get_resp(subject, stories)` to load brain recordings
3. **Load Text**: Use `get_story_wordseqs(stories)` to load word sequences with timing
4. **Extract Features**: Use `get_stim(stories, features)` to convert text to features
5. **Verify Alignment**: Check that dimensions match after accounting for trimming

The key insight is that the system maintains temporal alignment between:
- Words in the story (discrete events with precise timing)
- fMRI TRs (regularly sampled brain measurements)
- Features (interpolated and delayed to match brain response)

For more details, see:
- Training script: `decoding/train_EM.py`
- Decoder script: `decoding/run_decoder.py`
- Utility functions: `decoding/utils_resp.py` and `decoding/utils_stim.py`
