import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoFeatureExtractor, Wav2Vec2Model, Wav2Vec2Config
from tqdm import tqdm

from ups_challenge.dataloaders.contrastive import build_contrastive_dataset, collate_contrastive

from dotenv import load_dotenv

load_dotenv()


class ContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss for anchor-positive-negative triplets."""
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, anchor_emb, positive_emb, negative_emb):
        """
        Args:
            anchor_emb: [B, D] anchor embeddings
            positive_emb: [B, D] positive embeddings
            negative_emb: [B, D] negative embeddings
        Returns:
            loss: scalar contrastive loss
        """
        # Normalize embeddings
        anchor_emb = F.normalize(anchor_emb, p=2, dim=1)
        positive_emb = F.normalize(positive_emb, p=2, dim=1)
        negative_emb = F.normalize(negative_emb, p=2, dim=1)
        
        # Compute similarities (cosine similarity after normalization)
        pos_sim = (anchor_emb * positive_emb).sum(dim=1) / self.temperature  # [B]
        neg_sim = (anchor_emb * negative_emb).sum(dim=1) / self.temperature  # [B]
        
        # InfoNCE loss: -log(exp(pos) / (exp(pos) + exp(neg)))
        # Stack similarities: [2, B], then logsumexp along first dim to get [B]
        similarities = torch.stack([pos_sim, neg_sim], dim=0)  # [2, B]
        loss = -pos_sim + torch.logsumexp(similarities, dim=0)  # [B]
        
        return loss.mean()


def extract_embeddings(model, feature_extractor, waveforms, device):
    """
    Extract pooled embeddings from wav2vec2 last_hidden_state.
    
    Args:
        model: wav2vec2 AutoModel
        feature_extractor: AutoFeatureExtractor
        input_values: List of numpy arrays (raw audio waveforms)
        attention_mask: Unused, kept for API consistency
        device: torch device
    Returns:
        embeddings: [B, D] pooled embeddings
    """
    # Preprocess with feature extractor
    inputs = feature_extractor(
        waveforms,
        sampling_rate=feature_extractor.sampling_rate,
        padding=True,
        return_tensors="pt",
    ).to(device)
    
    # Get last_hidden_state from model
    outputs = model(**inputs)
    
    # Extract last_hidden_state: [B, T', D] where T' is the sequence length after feature extraction
    embeddings = outputs.last_hidden_state.mean(dim=1)  # [B, D]
    
    return embeddings


def train_contrastive(
    model_name="facebook/wav2vec2-base-960h",
    batch_size=4,
    num_workers=1,
    num_epochs=1,
    learning_rate=1e-4,
    temperature=0.07,
    device=None,
    langs=None,
    index_path="./data/lid_index_train.pkl",
    hf_token=None,
    max_steps=None,
):
    """
    Train wav2vec2 model with contrastive learning.
    
    Args:
        model_name: HuggingFace model name
        batch_size: Batch size for training
        num_workers: Number of dataloader workers
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        temperature: Temperature for contrastive loss
        device: torch device (auto-detected if None)
        langs: List of language codes to use (None for all)
        index_path: Path to language ID index
        hf_token: HuggingFace token
        max_steps: Maximum number of training steps (None for unlimited)
    """
    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Load model and feature extractor
    print(f"Loading model: {model_name}")
    config = Wav2Vec2Config(apply_spec_augment=False)
    model = Wav2Vec2Model.from_pretrained(model_name, config=config)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model.to(device)
    model.train()
    
    # Setup loss and optimizer
    criterion = ContrastiveLoss(temperature=temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Build dataset and dataloader
    print("Building contrastive dataset...")
    dataset = build_contrastive_dataset(
        langs=langs,
        index_path=index_path,
        hf_token=hf_token,
    )
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_contrastive,
    )
    
    # Training loop
    print("Starting training...")
    global_step = 0
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            if batch is None:
                continue
            
            # Extract anchor, positive, negative (as lists of numpy arrays)
            # FeatureExtractor will handle padding and attention masks
            anchor_inputs = batch["anchor"]  # List of numpy arrays
            positive_inputs = batch["positive"]  # List of numpy arrays
            negative_inputs = batch["negative"]  # List of numpy arrays
            
            # Extract embeddings (FeatureExtractor handles padding/masking internally)
            anchor_emb = extract_embeddings(
                model, feature_extractor, anchor_inputs, device
            )
            positive_emb = extract_embeddings(
                model, feature_extractor, positive_inputs, device
            )
            negative_emb = extract_embeddings(
                model, feature_extractor, negative_inputs, device
            )
            
            # Compute contrastive loss
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item(), "avg_loss": epoch_loss / num_batches})
            
            # Check max_steps
            if max_steps is not None and global_step >= max_steps:
                print(f"\nReached max_steps ({max_steps}), stopping training.")
                return
        
        print(f"Epoch {epoch+1} completed. Average loss: {epoch_loss / num_batches:.4f}")
    
    print("Training completed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train wav2vec2 with contrastive learning")
    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-base-960h",
                        help="HuggingFace model name")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of dataloader workers")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature for contrastive loss")
    parser.add_argument("--langs", type=str, nargs="*", default=None,
                        help="Language codes to use (empty for all)")
    parser.add_argument("--index_path", type=str, default="./data/lid_index.pkl",
                        help="Path to language ID index")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Maximum number of training steps")
    
    args = parser.parse_args()
    
    # Get HF token from env if not provided
    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    
    train_contrastive(
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        langs=args.langs,
        index_path=args.index_path,
        hf_token=hf_token,
        max_steps=args.max_steps,
    )
