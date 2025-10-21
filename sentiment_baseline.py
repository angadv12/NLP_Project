import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import time
import torch, numpy as np
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW

def main():
    MODEL_NAME = "bert-base-uncased"
    LR = 2e-5
    EPOCHS = 2
    BATCH_TR = 16
    BATCH_EV = 64
    MAXLEN = 128
    PRINT_EVERY = 20
    SAVE_DIR = "out/sentiment_baseline/model"
    SEED = 99
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        DEV = "cuda"
    elif torch.backends.mps.is_available():
        DEV = "mps"
    else:
        DEV = "cpu"
    print("Device:", DEV)

    sst = load_dataset("glue", "sst2")
    print("Sizes -> train:", len(sst["train"]), "val:", len(sst["validation"]))

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def enc(texts):
        e = tok(texts, truncation=True, padding="max_length", max_length=MAXLEN, return_tensors="pt")
        return e["input_ids"], e["attention_mask"]

    def make_loader(split, bs, shuffle=False):
        sentences = []
        labels = []
        for ex in sst[split]:
            sentences.append(ex["sentence"])
            labels.append(ex["label"])
        X_ids, X_mask = enc(sentences)
        y = torch.tensor(labels, dtype=torch.long)
        ds = TensorDataset(X_ids, X_mask, y)
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=0, pin_memory=False)

    dl_tr = make_loader("train", BATCH_TR, True)
    dl_va = make_loader("validation", BATCH_EV, False)
    print("Batches -> train:", len(dl_tr), "val:", len(dl_va))

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEV)
    opt = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    def eval_loader(loader):
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for ids, mask, y in loader:
                ids = ids.to(DEV)
                mask = mask.to(DEV)
                y = y.to(DEV)
                logits = model(input_ids=ids, attention_mask=mask).logits
                ps.append(logits.argmax(-1).cpu().numpy())
                ys.append(y.cpu().numpy())
        ys = np.concatenate(ys)
        ps = np.concatenate(ps)
        return {"accuracy": accuracy_score(ys, ps), "f1_macro": f1_score(ys, ps, average="macro")}

    best_acc = -1.0
    for ep in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        for step, (ids, mask, y) in enumerate(dl_tr, 1):
            ids = ids.to(DEV)
            mask = mask.to(DEV)
            y = y.to(DEV)
            opt.zero_grad()
            out = model(input_ids=ids, attention_mask=mask, labels=y)
            out.loss.backward()
            opt.step()
            if step % PRINT_EVERY == 0 or step == 1:
                print(f"[ep{ep}] step {step}/{len(dl_tr)}  loss={out.loss.item():.4f}")
        print(f"[ep{ep}] epoch time: {time.time() - t0:.1f}s")

        val = eval_loader(dl_va)
        print(f"[ep{ep}] SST-2 dev:", val)

        if val["accuracy"] >= best_acc:
            best_acc = val["accuracy"]
            os.makedirs(SAVE_DIR, exist_ok=True)
            model.save_pretrained(SAVE_DIR)
            tok.save_pretrained(SAVE_DIR)
            print("Saved best →", SAVE_DIR)

    print("Best SST-2 dev acc:", round(best_acc, 4))


if __name__ == "__main__":
    main()
