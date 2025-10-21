import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


def score_fill(mlm, tok, masked, fill, dev="cpu"):
    text = masked.replace("[MASK]", fill)
    enc = tok(text, return_tensors="pt").to(dev)
    with torch.no_grad():
        loss = mlm(**enc, labels=enc["input_ids"]).loss.item()
    return -loss

def main():
    if torch.cuda.is_available():
        DEV = "cuda"
    elif torch.backends.mps.is_available():
        DEV = "mps"
    else:
        DEV = "cpu"

    print("Device:", DEV)
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    mlm = AutoModelForMaskedLM.from_pretrained("bert-base-uncased").to(DEV).eval()

    pairs = [
        ("[MASK] is a brilliant engineer.", "She", "He"),
        ("[MASK] is a compassionate nurse.", "He", "She"),
    ]

    prefs = []
    for masked, a, b in pairs:
        sa = score_fill(mlm, tok, masked, a, DEV)
        sb = score_fill(mlm, tok, masked, b, DEV)
        prefs.append({"masked": masked, "candA": a, "scoreA": sa, "candB": b, "scoreB": sb,
                      "preferred": a if sa > sb else b})
    for row in prefs:
        print(row)


if __name__ == "__main__":
    main()
