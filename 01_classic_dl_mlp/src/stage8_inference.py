# 01_classic_dl_mlp/src/stage8_inference.py
# This script performs inference using the best saved MLP model on a batch of test data.
# It measures and reports the inference time per batch and per sample, and displays example predictions.

import time
import torch
from stage1_data_load import get_dataloaders, get_device, CLASSES
from stage4_model_baseline import MLP

# Main inference function
@torch.no_grad()
def main():
    device = get_device()
    _, _, test_loader = get_dataloaders(batch_size=256)

    # Load best model checkpoint
    ckpt = torch.load("outputs/models/best_model.pt", map_location=device)
    config = ckpt["config"]

    # Instantiate model and load state dict
    model = MLP(
        hidden1=config["hidden1"],
        hidden2=config["hidden2"],
        dropout=config["dropout"],
        use_bn=config["use_bn"]
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Get a batch of test data
    x, y = next(iter(test_loader))
    x = x.to(device)

    # warm-up for accurate timing 
    _ = model(x)
    
    # Measure inference time
    t0 = time.perf_counter()
    logits = model(x)
    t1 = time.perf_counter()
    
    # Get predictions 
    preds = logits.argmax(dim=1).cpu()

    # Calculate timing metrics 
    batch_ms = (t1 - t0) * 1000
    per_sample_ms = batch_ms / x.size(0)

    print("✅ STAGE 8 — Inference")
    print(f"Batch inference: {batch_ms:.2f} ms for {x.size(0)} samples")
    print(f"Per-sample: {per_sample_ms:.4f} ms")
    print("Example predictions:", [CLASSES[p.item()] for p in preds[:10]])

if __name__ == "__main__":
    main()
