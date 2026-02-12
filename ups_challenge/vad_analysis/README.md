# Core Components

1. `vad_lookup.py` - VAD Index Builder

Input: `vad_results.jsonl` (contains VAD timestamps for all 2.1M audio files)

Process:

* Parses JSONL line by line with progress bar
* Extracts speech segments with start/end timestamps
* Converts sample-based timestamps to seconds
* Organizes by tar file for efficient lookup
* Output: Sharded pickle files (000001.pkl, 000002.pkl, etc.)
* Each shard contains: {filename: {"segments": [(start, end), ...], "duration": X}}
* Memory Optimization: Flushes to disk every 10,000 lines to manage memory

2. `apply_vad.py` - VAD-Enhanced Audio Processor. Key Functions:`decode_and_normalize_with_vad()`

Process: 

* Parse URL: Extract tar number from URL pattern
* Language Filter: Skip if language not in desired_languages
* VAD Lookup: Load pre-processed speech segments for this tar file
* Segment Selection: Find speech segments ≥10 seconds
* Chunk Generation: Extract random 10-second chunks from valid segments
* Batching: Stack chunks into tensor with attention masks

---

**Performance Features:**

* `@lru_cache(maxsize=16)`: Caches recently loaded VAD shards for speed

* Progressive Sampling: Extracts up to max_chunks_per_example from each file

* Efficient Decoding: Uses torchcodec.decoders.AudioDecoder with range extraction

* `collate_fn()`
    Purpose: Combines multiple samples into a training batch

    Handles: Variable number of chunks per audio file

    Output:

    `input_values`: Concatenated audio tensors [total_chunks, 160000]

    `attention_mask`: Ones tensor matching input shape

    `language`: List of language codes for each chunk

---

**`vad_lookup.py` Helper Functions**

* `_parse_speech_segments()` converts VAD timestamps from sample indices to seconds and filters invalid/missing timestamps. Returns None if no speech detected

* `_flush_shards()` implements atomic write pattern with temp files. Merges new data with existing shards. Prevents data corruption during interrupted runs

---

**Skip Conditions:**

* No VAD data available
* No speech segments detected
* No segments ≥10 seconds long
* Language not in target list

---
**The usage example is `main_example.py`**

```
batch_size = 1  # Controls how many audio FILES are processed per batch
```

* Purpose: Determines how many audio files (from the WebDataset) are processed together in a single batch

* Effect: Each audio file can produce multiple 10-second chunks (up to max_chunks_per_example in apply_vad.py)

* Example: With batch_size=2 and max_chunks_per_example=3, you could get 2-6 chunks per batch (depending on available speech segments)

```
max_batches = 20  # Limits total number of batches returned
```

* Purpose: Safety limit to prevent infinite loops

* Implementation: Breaks the loop after processing specified number of batches