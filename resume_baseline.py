import pandas as pd, numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from utils import set_seed, simple_metrics

def to_hf(df, tok, max_len=160):
    ds = Dataset.from_pandas(df[["text", "label", "gender"]].reset_index(drop=True))

    def _tok(batch):
        return tok(batch["text"], truncation=True, padding="max_length", max_length=max_len)

    ds = ds.map(_tok, batched=True)
    cols_to_keep = ["input_ids", "attention_mask", "label", "gender"]
    cols_to_remove = []
    for c in ds.column_names:
        if c not in cols_to_keep:
            cols_to_remove.append(c)
    ds = ds.remove_columns(cols_to_remove)
    ds = ds.with_format("torch")
    return ds

def main():
    set_seed(99)
    MODEL = "bert-base-uncased"
    CSV = "data/bios_subset.csv"

    df = pd.read_csv(CSV)
    occ_labels = sorted(df["occupation"].unique())
    occ2id = {}
    for i, o in enumerate(occ_labels):
        occ2id[o] = i
    df["label"] = df["occupation"].map(occ2id)

    train = df.sample(frac=0.8, random_state=99)
    tmp = df.drop(train.index)
    valid = tmp.sample(frac=0.5, random_state=99)
    test = tmp.drop(valid.index)

    tok = AutoTokenizer.from_pretrained(MODEL)
    ds_train = to_hf(train, tok)
    ds_val = to_hf(valid, tok)
    ds_test = to_hf(test, tok)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(occ_labels))

    args = TrainingArguments(
        output_dir="out/bios_baseline",
        logging_steps=50,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=2,
        weight_decay=0.01,
        seed=99
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tok,
        compute_metrics=simple_metrics,
    )

    print("Starting training...\n")
    trainer.train()
    print("\nBIOS valid metrics:", trainer.evaluate())

    trainer.save_model("out/bios_baseline/model")
    tok.save_pretrained("out/bios_baseline/model")
    test[["text", "label", "gender"]].to_csv("out/bios_baseline/test_cache.csv", index=False)
    with open("out/bios_baseline/occ_labels.txt", "w") as f:
        for o in occ_labels:
            f.write(o + "\n")

    print("Saved model and test cache to: out/bios_baseline/model\n")

if __name__ == "__main__":
    main()
