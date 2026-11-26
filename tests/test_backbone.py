from pathlib import Path

import torch
from tokenizers import Tokenizer

from config import ModelConfig
from quillinstruct.backbone.gpt import GPT

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts" / "base"

def test_backbone() -> None:
    config = ModelConfig()

    print("Loading model...")
    model = GPT(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        max_context=config.max_context,
        feedforward_dim=config.feedforward_dim,
        dropout=config.dropout
    )
    print("Model loaded")

    print("Loading weights...")
    state_dict = torch.load(
        BASE / "quillGPT_v0.4.0_best.pth",
        map_location="cpu",
        weights_only=True,
    )
    print("Weights loaded")

    model.load_state_dict(state_dict, strict=True)

    del state_dict

    parameter_count = sum(p.numel() for p in model.parameters())
    
    assert parameter_count == 235_082_752
    
    print("Loading tokenizer...")
    tokenizer: Tokenizer = Tokenizer.from_file(str(BASE / "bpe_tokenizer_16384.json"))
    print("Tokenizer loaded")

    assert tokenizer.get_vocab_size() == config.vocab_size

    eos_id = tokenizer.token_to_id("<eos>")
    assert eos_id is not None

    token_ids = tokenizer.encode("The Earth revolves around").ids
    tokens = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)

    model.eval()

    with torch.no_grad():
        logits = model(tokens)

    assert logits.shape == (1, tokens.shape[1], config.vocab_size)
    assert torch.isfinite(logits).all()
    assert not logits.requires_grad

    print(f"Parameters: {parameter_count:,}")
    print(f"Input shape: {tokens.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"EOS token ID: {eos_id}")
