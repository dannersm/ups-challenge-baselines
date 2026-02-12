import re
import os
import torch
import pickle
from pathlib import Path
from functools import lru_cache
from torchcodec.decoders import AudioDecoder
import random

@lru_cache(maxsize=16)
def load_vad_shard(tar_number: str, base_dir="./data/vad_shards"):
    path = Path(base_dir) / f"{tar_number}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"VAD shard not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

_tar_re = re.compile(r"/(audio2|audio)/(\d+)\.tar")

def _parse_tar_number_from_url(url: str):
    m = _tar_re.search(url)
    if m:
        return m.group(2).zfill(6)
    return None

def decode_and_normalize_with_vad(sample, desired_languages: list[str] = None, tar_to_lang: dict = None,  
    target_sr: int = 16_000, chunk_sec: float = 10.0, max_chunks_per_example: int = 16, shuffle_chunks: bool = False,
    debug_counter=[0]):

    debug_counter[0] += 1
    if debug_counter[0] % 100 == 0:
        print(f"[VAD] Processed {debug_counter[0]} samples")
    
    mp3_bytes, key, url = sample
    
    filename = os.path.basename(key)
    if not filename.endswith(".mp3"):
        filename += ".mp3"
    
    tar_number = _parse_tar_number_from_url(url)
    if not tar_number:
        return None
    
    lookup_key = (tar_number, filename)
    language = tar_to_lang.get(lookup_key, "unknown") if tar_to_lang else "unknown"
    
    # filter by desired languages
    if desired_languages and language not in desired_languages:
        return None
    
    vad_shard = load_vad_shard(tar_number)
    vad_data = vad_shard.get(filename)
    
    if not vad_data:
        print(f"[VAD] SKIPPING - No VAD data for ({tar_number}, {filename})")
        return None
    
    segments = vad_data.get('segments', [])
    
    if not segments:
        print(f"[VAD] SKIPPING - No segments in VAD data for ({tar_number}, {filename})")
        return None
    
    decoder = AudioDecoder(source=mp3_bytes, sample_rate=target_sr, num_channels=1)
    chunk_samples = int(chunk_sec * target_sr)
    
    def _stream_chunk(start_sec, end_sec):
        samples = decoder.get_samples_played_in_range(start_sec, end_sec)
        chunk = samples.data.squeeze(0)
        if chunk.shape[-1] < chunk_samples:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_samples - chunk.shape[-1]))
        return chunk
    
    candidate_windows = []
    for seg_start, seg_end in segments:
        if seg_end - seg_start >= chunk_sec:
            candidate_windows.append((seg_start, seg_end - chunk_sec))

    if not candidate_windows:
        print(f"[VAD] SKIPPING - No long enough segments (>={chunk_sec}s) for ({tar_number}, {filename})")
        return None

    max_possible_chunks = 0
    for seg_start, max_start in candidate_windows:
        seg_len = max_start - seg_start
        possible = max(1, int(seg_len // chunk_sec))
        max_possible_chunks += possible

    num_chunks = min(max_chunks_per_example, max_possible_chunks)

    if num_chunks <= 0:
        return None

    output_chunks = []
    for _ in range(num_chunks):
        seg_start, max_start = random.choice(candidate_windows)
        start_sec = random.uniform(seg_start, max_start)
        end_sec = start_sec + chunk_sec
        output_chunks.append(_stream_chunk(start_sec, end_sec))
    
    if shuffle_chunks:
        random.shuffle(output_chunks)
    
    batch_wave = torch.stack(output_chunks)
    attention_mask = torch.ones_like(batch_wave, dtype=torch.long)
    
    return {
        "input_values": batch_wave,
        "attention_mask": attention_mask,
        "language": language,  
    }

def collate_fn(batch: list):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    
    input_values = torch.cat([sample["input_values"] for sample in batch], dim=0)
    attention_mask = torch.cat([sample["attention_mask"] for sample in batch], dim=0)
    
    # Since all chunks from same file have same language, just take first one from each sample
    languages = []
    for sample in batch:
        num_chunks = sample["input_values"].shape[0]
        languages.extend([sample["language"]] * num_chunks)
        
    return {
        "input_values": input_values, 
        "attention_mask": attention_mask,
        "language": languages  
    }