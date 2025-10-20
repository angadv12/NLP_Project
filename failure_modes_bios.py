import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix
from utils import device


def predict_occ(model, tok, ds_texts, batch=128, dev="cpu"):
    preds = []
    model.eval()
    for i in range(0, len(ds_texts), batch):
        enc = tok(ds_texts[i:i + batch], truncation=True, padding=True, return_tensors="pt").to(dev)
        with torch.no_grad():
            logits = model(**enc).logits
        preds.extend(logits.argmax(-1).cpu().numpy().tolist())
    return np.array(preds)


def error_rates_by_group(y, yhat, g, n_labels):
    res = {}
    for group in sorted(set(g)):
        mask = (g == group)
        cm = confusion_matrix(y[mask], yhat[mask], labels=list(range(n_labels)))
        fprs, fnrs = [], []
        for c in range(n_labels):
            TP = cm[c, c]
            FN = cm[c, :].sum() - TP
            FP = cm[:, c].sum() - TP
            TN = cm.sum() - (TP + FN + FP)
            fpr = FP / (FP + TN + 1e-9)
            fnr = FN / (FN + TP + 1e-9)
            fprs.append(fpr);
            fnrs.append(fnr)
        res[group] = {
            "ACC": float((yhat[mask] == y[mask]).mean()),
            "FPR_macro": float(np.mean(fprs)),
            "FNR_macro": float(np.mean(fnrs))
        }
    accs = []
    fprs = []
    fnrs = []
    for v in res.values():
        accs.append(v["ACC"])
        fprs.append(v["FPR_macro"])
        fnrs.append(v["FNR_macro"])

    res["gap_ACC"] = float(max(accs) - min(accs))
    res["gap_FPR"] = float(max(fprs) - min(fprs))
    res["gap_FNR"] = float(max(fnrs) - min(fnrs))
    return res


def sample_failures(df_texts, y, yhat, g, label_names, k=3):
    rows = []
    for group in sorted(set(g)):
        mask = (g == group) & (yhat != y)
        idxs = np.where(mask)[0][:k]
        for i in idxs:
            rows.append({"group": group,
                         "gold": label_names[y[i]],
                         "pred": label_names[yhat[i]],
                         "text": df_texts.iloc[i][:160] + "..."})
    return rows


def main():
    DEV = device()
    MODEL_DIR = "out/bios_baseline/model"
    TEST_CACHE = "out/bios_baseline/test_cache.csv"
    LABELS_TXT = "out/bios_baseline/occ_labels.txt"

    test_df = pd.read_csv(TEST_CACHE)
    label_names = []
    with open(LABELS_TXT) as f:
        for l in f.read().splitlines():
            l = l.strip()
            if l:
                label_names.append(l)

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEV)

    yhat = predict_occ(model, tok, test_df["text"].tolist(), dev=DEV)
    ytrue = test_df["label"].to_numpy()
    g = test_df["gender"].to_numpy()

    errs = error_rates_by_group(ytrue, yhat, g, n_labels=len(label_names))
    print("Error-rate and accuracy gaps:", errs)

    fails = sample_failures(test_df["text"], ytrue, yhat, g, label_names, k=3)
    print("\nSample misclassifications (qualitative):")
    for r in fails:
        print(r)


if __name__ == "__main__":
    main()
