import random
import webdataset as wds
from torchcodec.decoders import AudioDecoder
from .urls import build_urls


def decode_contrastive(sample, target_sr=16000, chunk_sec=10.0, min_offset_frac=0.1):
    """Decode one mp3 into two random chunks (anchor & positive) as raw audio arrays.
    
    No padding or masking is done here - the FeatureExtractor will handle that.
    Returns raw audio chunks as numpy arrays.
    """
    mp3_bytes, key, url = sample
    try:
        decoder = AudioDecoder(source=mp3_bytes, sample_rate=target_sr, num_channels=1)
        duration = decoder.metadata.duration_seconds_from_header
    except Exception:
        return None

    if duration < chunk_sec:
        return None  # too short even for one chunk

    max_start_sec = max(0.0, duration - chunk_sec)
    anchor_start = random.uniform(0.0, max_start_sec)
    offset = random.uniform(min_offset_frac * chunk_sec, max_start_sec)
    positive_start = (anchor_start + offset) % max_start_sec

    def extract(start_sec):
        """
        Return raw audio chunk as numpy array.
        Returns None if extraction fails (corrupted audio).
        """
        try:
            end_sec = start_sec + chunk_sec
            samples = decoder.get_samples_played_in_range(start_sec, end_sec)
            chunk = samples.data.squeeze(0)  # [T] or [1, T] -> [T]
            
            # Convert to numpy array (FeatureExtractor expects numpy)
            chunk_np = chunk.cpu().numpy()
            return chunk_np
        except (RuntimeError, ValueError, Exception):
            # Handle corrupted audio files gracefully
            return None

    anchor_chunk = extract(anchor_start)
    positive_chunk = extract(positive_start)
    
    # If either extraction failed, return None to skip this sample
    if anchor_chunk is None or positive_chunk is None:
        return None

    return {
        "anchor": anchor_chunk,
        "positive": positive_chunk,
        "meta": {"key": key, "url": url, "duration": duration},
    }


def collate_contrastive(batch):
    """Build batch dict with anchor, positive, and negative sets as lists of arrays.
    
    Returns lists of numpy arrays (not stacked tensors). The FeatureExtractor
    will handle padding and attention masks during preprocessing.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    # Extract anchor and positive chunks (as numpy arrays)
    anchors = [b["anchor"] for b in batch]
    positives = [b["positive"] for b in batch]

    # negatives: randomly shuffle anchors to create negative pairs
    perm = list(range(len(anchors)))
    random.shuffle(perm)
    negatives = [anchors[i] for i in perm]

    return {
        "anchor": anchors,
        "positive": positives,
        "negative": negatives,
    }


def build_contrastive_dataset(
    langs=None,
    index_path="./data/lid_index.pkl",
    hf_token=None,
    target_sr=16000,
    chunk_sec=10.0,
):
    """Return a WebDataset yielding dicts with anchor/positive audio chunks (with masks)."""
    if langs is None:
        langs = []

    urls = build_urls(langs, index_path=index_path, hf_token=hf_token)

    dataset = (
        wds.WebDataset(
            urls, 
            shardshuffle=True,
            handler=wds.handlers.ignore_and_continue  # Handle network errors gracefully
        )
        .to_tuple("mp3", "__key__", "__url__", handler=wds.handlers.ignore_and_continue)
        .map(lambda s: decode_contrastive(s, target_sr=target_sr, chunk_sec=chunk_sec))
    )

    return dataset
