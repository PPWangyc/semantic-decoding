# Data Flow Diagram

This document provides a visual representation of how fMRI data and text stimuli flow through the semantic-decoding pipeline.

## Data Loading Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RAW DATA FILES                                    │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ├───────────────────────────────────┐
                                │                                   │
                                ▼                                   ▼
                ┌───────────────────────────┐     ┌──────────────────────────┐
                │   fMRI Response Data      │     │   Text Stimulus Data     │
                │   (.hf5 files)            │     │   (.TextGrid files)      │
                ├───────────────────────────┤     ├──────────────────────────┤
                │ Location:                 │     │ Location:                │
                │ train_response/           │     │ train_stimulus/          │
                │   [SUBJECT]/              │     │   [STORY].TextGrid       │
                │     [STORY].hf5           │     │                          │
                │                           │     │ Content:                 │
                │ Content:                  │     │ - Words                  │
                │ - Brain activity          │     │ - Word start times       │
                │ - Shape: [TRs, voxels]    │     │ - Word end times         │
                │ - TR ≈ 2 seconds          │     │ - Organized in tiers     │
                └───────────────────────────┘     └──────────────────────────┘
                                │                                   │
                                │                                   │
                                ▼                                   ▼
                ┌───────────────────────────┐     ┌──────────────────────────┐
                │   get_resp()              │     │   get_story_wordseqs()   │
                │   (utils_resp.py)         │     │   (utils_stim.py)        │
                └───────────────────────────┘     └──────────────────────────┘
                                │                                   │
                                │                                   │
                                ▼                                   ▼
                ┌───────────────────────────┐     ┌──────────────────────────┐
                │   Response Matrix         │     │   DataSequence Object    │
                │   numpy array             │     │                          │
                │   [TRs, voxels]           │     │   .data: word list       │
                │                           │     │   .data_times: word times│
                │   Example:                │     │   .tr_times: TR times    │
                │   [245, 52644]            │     │   .split_inds: TR splits │
                │   245 TRs                 │     │                          │
                │   52644 voxels            │     │   Example:               │
                │                           │     │   data: ["the", "cat"...]│
                │                           │     │   times: [0.2, 0.8, ...] │
                └───────────────────────────┘     └──────────────────────────┘
                                │                                   │
                                │                                   │
                                │                 ┌─────────────────┘
                                │                 │
                                │                 ▼
                                │     ┌──────────────────────────┐
                                │     │   GPT Feature Extractor  │
                                │     │   (LMFeatures)           │
                                │     └──────────────────────────┘
                                │                 │
                                │                 │
                                │                 ▼
                                │     ┌──────────────────────────┐
                                │     │   get_stim()             │
                                │     │   (utils_stim.py)        │
                                │     └──────────────────────────┘
                                │                 │
                                │                 │
                                │                 ▼
                                │     ┌──────────────────────────┐
                                │     │   Stimulus Features      │
                                │     │   [TRs, features]        │
                                │     │                          │
                                │     │   - Lanczos interpolated │
                                │     │   - Normalized           │
                                │     │   - Delayed (HRF)        │
                                │     └──────────────────────────┘
                                │                 │
                                └─────────────────┘
                                          │
                                          ▼
                        ┌─────────────────────────────────┐
                        │   ALIGNED DATA FOR MODELING     │
                        │                                 │
                        │   Response:  [TRs, voxels]      │
                        │   Stimulus:  [TRs, features]    │
                        │                                 │
                        │   Each row = same time point    │
                        └─────────────────────────────────┘
                                          │
                                          ▼
                        ┌─────────────────────────────────┐
                        │   Encoding Model Training       │
                        │   (train_EM.py)                 │
                        │                                 │
                        │   or                            │
                        │                                 │
                        │   Decoder Inference             │
                        │   (run_decoder.py)              │
                        └─────────────────────────────────┘
```

## Temporal Alignment

```
TIME AXIS (seconds)
────────────────────────────────────────────────────────────────────►

Words:
  ├─w1─┤ ├─w2──┤ ├─w3─┤   ├──w4───┤ ├─w5─┤   ├──w6───┤
  0.2   0.8     1.3     2.1        2.7     3.5        4.2

Word Times (centers):
  *       *       *        *         *        *
  0.5     1.0     1.5      2.4       3.1      3.9

TRs (acquisitions every ~2 seconds):
  │              TR1              │              TR2              │
  0.0                            2.0                            4.0

TR Times (centers):
              *                              *
              1.0                            3.0

Feature Extraction Process:
  1. Extract GPT features for each word
  2. Interpolate word features → TR features (Lanczos)
  3. Apply hemodynamic delays [1, 2, 3, 4] TRs
  4. Normalize features
  
Result: Stimulus features aligned to TR times with HRF delays
```

## Data Structure Details

### Response Dictionary Structure

```python
# When stack=False
resp_dict = {
    "story1": numpy.array([n_TRs_1, n_voxels]),  
    "story2": numpy.array([n_TRs_2, n_voxels]),
    ...
}

# When stack=True
resp_matrix = numpy.array([total_TRs, n_voxels])
# Where: total_TRs = n_TRs_1 + n_TRs_2 + ...
```

### DataSequence Structure

```python
ds = DataSequence(
    data=["word1", "word2", ...],           # List of words
    split_inds=[34, 71, 105, ...],          # Indices where TRs split
    data_times=[0.5, 1.0, 1.5, ...],        # Word center times
    tr_times=[1.0, 3.0, 5.0, ...]           # TR acquisition times
)

# Access patterns:
words = ds.data                  # All words
word_times = ds.data_times       # Timing for each word
tr_times = ds.tr_times           # TR acquisition times
chunks = ds.chunks()             # Words grouped by TR
```

## Processing Pipeline Example

### Training Pipeline

```
1. Load Stories
   └─> sess_to_story.json → story list

2. Load fMRI Data
   └─> get_resp(subject, stories)
       └─> [story1.hf5, story2.hf5, ...] → stacked matrix

3. Load Text Data  
   └─> get_story_wordseqs(stories)
       └─> [story1.TextGrid, story2.TextGrid, ...] → DataSequence objects

4. Extract Features
   └─> get_stim(stories, gpt_features)
       ├─> GPT embeddings for each word
       ├─> Lanczos interpolation to TR times
       ├─> Normalization (z-score)
       └─> Apply delays [1, 2, 3, 4]

5. Train Model
   └─> bootstrap_ridge(stimulus, response)
       └─> Encoding model weights
```

### Test/Decode Pipeline

```
1. Load Test Data
   ├─> test_response/[subject]/[experiment]/[task].hf5
   └─> Response matrix: [n_TRs, n_voxels]

2. Load Models
   ├─> encoding_model_[gpt].npz
   │   ├─> weights
   │   ├─> noise_model
   │   ├─> voxels
   │   └─> tr_stats
   └─> word_rate_model_[type].npz
       ├─> weights
       ├─> voxels
       └─> mean_rate

3. Predict Word Times
   └─> predict_word_rate() → predict_word_times()
       └─> Estimated word timing from brain activity

4. Decode Words
   └─> Decoder with:
       ├─> Language model (GPT)
       ├─> Encoding model (predict brain from words)
       └─> Beam search over word sequences
```

## Key Configuration Parameters

```python
# From config.py

TRIM = 5                      # TRs trimmed from start/end
STIM_DELAYS = [1, 2, 3, 4]   # Hemodynamic response delays
RESP_DELAYS = [-4, -3, -2, -1] # For word rate prediction
VOXELS = 10000                # Top voxels to use
GPT_LAYER = 9                 # Which GPT layer for features
GPT_WORDS = 5                 # Context window size
```

## File Format Specifications

### .hf5 (HDF5) Format
```
HDF5 File Structure:
├─ "data": Dataset
   ├─ dtype: float32 or float64
   ├─ shape: (n_TRs, n_voxels)
   └─ NaN values are replaced with 0
```

### .TextGrid Format
```
Praat TextGrid File:
├─ Tier 0: Phonemes
├─ Tier 1: Words ← Used by default
├─ Tier 2: Characters
└─ Tier 3: Dialogue

Each entry has:
├─ xmin: start time
├─ xmax: end time
└─ text: content
```

## Common Transformations

### Word-Level → TR-Level

```
Words (irregular timing)  →  TRs (regular sampling)
        ↓
    Lanczos Interpolation
        ↓
    TR-aligned features
```

### Normalization

```
Features  →  Z-score normalization  →  Standardized features
             (per feature dimension)
```

### Delay Application

```
Features[t] → [
    Features[t+1],  # delay 1
    Features[t+2],  # delay 2
    Features[t+3],  # delay 3
    Features[t+4]   # delay 4
] concatenated
```

This accounts for hemodynamic response function (HRF) delay between stimulus and brain response.
