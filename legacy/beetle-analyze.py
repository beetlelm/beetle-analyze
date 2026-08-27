#!/usr/bin/env python3
import os
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfApi

# ---------------------------------------------------------
# 1. Configuration & Setup
# ---------------------------------------------------------

# Set to "hf" to pull from BeetleLM Hub or "local" for local runs
MODE = "hf" 

# HF Hub Settings
HF_USER = "BeetleLM"
HF_MODELS = ["beetlelm_eng-nld_heritage", "beetlelm_eng-nld_late"]

# Local Settings
LOCAL_CHECKPOINT_DIR = Path("./runs/beetlelm_zho-ukr_heritage")
LOCAL_GLOB = "checkpoint_*"

# Evaluation Data
DATA_DIR = Path("./datasets")
L1_FILE = DATA_DIR / "eng_eval.txt"
L2_FILE = DATA_DIR / "nld_eval.txt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "./hf_cache" # Prevents re-downloading 665MB bins constantly

# ---------------------------------------------------------
# 2. Analysis Modules
# ---------------------------------------------------------

def centered_kernel_alignment(X, Y):
    """Linear CKA to measure representational similarity."""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    dot_prod = (X.T @ Y).norm()**2
    norm_x = (X.T @ X).norm()**2
    norm_y = (Y.T @ Y).norm()**2
    return (dot_prod / (torch.sqrt(norm_x * norm_y))).item()

class LanguageProbe(nn.Module):
    """Simple linear probe to check language separation."""
    def __init__(self, dim):
        super().__init__()
        self.classifier = nn.Linear(dim, 2)
    def forward(self, x):
        return self.classifier(x)

def run_language_probe(l1_states, l2_states):
    """Trains a probe to distinguish L1 vs L2 hidden states."""
    dim = l1_states.shape[-1]
    probe = LanguageProbe(dim).to(DEVICE)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Labeling: L1=0, L2=1
    X = torch.cat([l1_states, l2_states], dim=0).to(DEVICE)
    y = torch.cat([torch.zeros(len(l1_states)), torch.ones(len(l2_states))], dim=0).long().to(DEVICE)
    
    # Quick Training (5 epochs)
    for _ in range(5):
        optimizer.zero_grad()
        loss = criterion(probe(X), y)
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        accuracy = (probe(X).argmax(1) == y).float().mean().item()
    return accuracy

# ---------------------------------------------------------
# 3. Model Processing
# ---------------------------------------------------------

def evaluate_model(model, tokenizer, file_path, seq_len=512):
    """Computes PPL and extracts hidden states for the last token."""
    if not file_path.exists():
        return None, None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(DEVICE)
    
    losses = []
    hidden_states = []
    
    for i in range(0, input_ids.size(1) - seq_len, seq_len):
        chunk = input_ids[:, i:i+seq_len]
        with torch.no_grad():
            outputs = model(chunk, labels=chunk, output_hidden_states=True)
            losses.append(outputs.loss.item())
            # Extract last layer hidden state of the last token
            hidden_states.append(outputs.hidden_states[-1][:, -1, :].cpu())
            
    avg_loss = np.mean(losses)
    return np.exp(avg_loss), torch.cat(hidden_states, dim=0)

def get_tasks():
    tasks = []
    if MODE == "local":
        for p in sorted(LOCAL_CHECKPOINT_DIR.glob(LOCAL_GLOB)):
            tasks.append({"id": p.name, "path": str(p), "rev": None})
    else:
        api = HfApi()
        for m in HF_MODELS:
            repo_id = f"{HF_USER}/{m}"
            info = api.repo_info(repo_id)
            branches = [b.name for b in info.refs if b.name.startswith("step-")]
            branches.sort(key=lambda x: int(x.split("-")[1]))
            for b in branches:
                tasks.append({"id": f"{m}_{b}", "path": repo_id, "rev": b})
    return tasks

# ---------------------------------------------------------
# 4. Main Execution
# ---------------------------------------------------------

def main():
    tasks = get_tasks()
    results = []

    for task in tqdm(tasks, desc="Processing BeetleLM Branches"):
        # Load
        model = AutoModelForCausalLM.from_pretrained(
            task["path"], revision=task["rev"], cache_dir=CACHE_DIR
        ).to(DEVICE).eval()
        tokenizer = AutoTokenizer.from_pretrained(task["path"], revision=task["rev"])

        # Eval L1 & L2
        l1_ppl, l1_hidden = evaluate_model(model, tokenizer, L1_FILE)
        l2_ppl, l2_hidden = evaluate_model(model, tokenizer, L2_FILE)

        # Cross-lingual Metrics
        min_n = min(len(l1_hidden), len(l2_hidden))
        cka_score = centered_kernel_alignment(l1_hidden[:min_n], l2_hidden[:min_n])
        probe_acc = run_language_probe(l1_hidden, l2_hidden)

        results.append({
            "step": task["id"],
            "l1_ppl": l1_ppl,
            "l2_ppl": l2_ppl,
            "cka_sim": cka_score,
            "lang_separation": probe_acc
        })

        # Cleanup
        del model, tokenizer
        torch.cuda.empty_cache()

    # Save Results
    with open("beetlelm_final_analysis.json", "w") as f:
        json.dump(results, f, indent=4)

    # Simple Plotting
    sns.set_style("darkgrid")
    steps = [r["step"] for r in results]
    plt.figure(figsize=(10, 5))
    plt.plot(steps, [r["l1_ppl"] for r in results], label="L1 PPL", marker='o')
    plt.plot(steps, [r["l2_ppl"] for r in results], label="L2 PPL", marker='s')
    plt.title("BeetleLM Curriculum Learning Dynamics")
    plt.xticks(rotation=45)
    plt.legend()
    plt.savefig("learning_curves.png")
    print("Analysis complete. Results saved to JSON and learning_curves.png")

if __name__ == "__main__":
    main()
