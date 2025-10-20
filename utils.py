import os, random, numpy as np, torch
from sklearn.metrics import accuracy_score, f1_score
from collections import defaultdict

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def set_seed(seed: int = 99):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed)


def device():
    if torch.cuda.is_available():
        return "cuda"  
    else:
        return "cpu"


def simple_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1}



def group_gap(y_true, y_pred, groups):
    buckets = defaultdict(lambda: {"y": [], "p": []})
    for yt, yp, g in zip(y_true, y_pred, groups):
        buckets[g]["y"].append(yt)
        buckets[g]["p"].append(yp)
    out = {}
    for g, d in buckets.items():
        out[g] = {
            "acc": accuracy_score(d["y"], d["p"]),
            "f1_macro": f1_score(d["y"], d["p"], average="macro"),
        }
    if len(out) >= 2:
        accs = []
        f1s = []
        for v in out.values():
            accs.append(v["acc"])
            f1s.append(v["f1_macro"])
        out["gap_accuracy"] = float(max(accs) - min(accs))
        out["gap_f1_macro"] = float(max(f1s) - min(f1s))
    else:
        out["gap_accuracy"] = 0.0
        out["gap_f1_macro"] = 0.0
    return out
