import pandas as pd, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import device, group_gap


def batched_preds(model, tok, texts, batch=64, dev="cpu"):
    preds = []
    model.eval()
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], truncation=True, padding=True, return_tensors="pt").to(dev)
        with torch.no_grad():
            logits = model(**enc).logits
        preds.extend(logits.argmax(-1).cpu().numpy().tolist())
    return np.array(preds)

def main():
    DEV = device()
    MODEL_DIR = "out/sentiment_baseline/model"
    EEC_PATH = "data/eec_proxy.csv"

    df = pd.read_csv(EEC_PATH)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEV)

    y_pred = batched_preds(model, tok, df["text"].tolist(), dev=DEV)
    y_true = df["label"].tolist()
    groups = df["group"].tolist()

    print("EEC group metrics + gaps:", group_gap(y_true, y_pred, groups))


if __name__ == "__main__":
    main()
