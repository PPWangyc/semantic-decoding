# Quick Reference: Loading fMRI and Text Data

## Quick Start

```python
from decoding.utils_resp import get_resp
from decoding.utils_stim import get_story_wordseqs

# Load fMRI data
subject = "UTS03"
stories = ["alternateithicatom"]
fmri_data = get_resp(subject, stories, stack=True)
# Returns: [n_TRs, n_voxels] numpy array

# Load text with timing
text_data = get_story_wordseqs(stories)
# Returns: dict with DataSequence objects
# Access: text_data["alternateithicatom"].data  # list of words
#         text_data["alternateithicatom"].data_times  # word times
```

## Core Functions

| Function | File | Purpose | Returns |
|----------|------|---------|---------|
| `get_resp()` | `utils_resp.py` | Load fMRI responses | `[TRs, voxels]` array |
| `get_story_wordseqs()` | `utils_stim.py` | Load word sequences | Dict of DataSequence |
| `get_stim()` | `utils_stim.py` | Extract features | `[TRs, features]` array |

## Data Files

- **fMRI responses**: `data_train/train_response/[SUBJECT]/[STORY].hf5`
- **Text stimuli**: `data_train/train_stimulus/[STORY].TextGrid`
- **Metadata**: `data_train/respdict.json`, `sess_to_story.json`

## Example: Load and Inspect

```python
import numpy as np
from decoding.utils_resp import get_resp
from decoding.utils_stim import get_story_wordseqs

# Load data
subject, story = "UTS03", "alternateithicatom"
fmri = get_resp(subject, [story], stack=True)
text = get_story_wordseqs([story])[story]

# Inspect
print(f"fMRI shape: {fmri.shape}")
print(f"Words: {len(text.data)}")
print(f"First words: {text.data[:5]}")
print(f"Word times: {text.data_times[:5]}")
```

## DataSequence Object

```python
ds = text_data["story_name"]
ds.data         # List of words
ds.data_times   # Word center times (seconds)
ds.tr_times     # TR acquisition times (seconds)
ds.split_inds   # Indices splitting words into TRs
```

For detailed explanation, see [LOADING_DATA_TUTORIAL.md](LOADING_DATA_TUTORIAL.md)
