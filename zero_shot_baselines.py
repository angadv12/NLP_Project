import pandas as pd
from datasets import load_dataset
from transformers import pipeline

def main():
    sst = load_dataset("glue", "sst2")
    sst_texts = []
    for i in [0, 10, 20, 30, 40]:
        sst_texts.append(sst["validation"][i]["sentence"])

    bios = pd.read_csv("data/bios_subset.csv")
    occ_labels = sorted(bios["occupation"].unique())
    bios_texts = bios["text"].tolist()[:8]

    clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
    zs_sent = []
    for x in sst_texts:
        zs_sent.append(clf(x, candidate_labels=["positive", "negative"]))

    zs_bios = []
    for x in bios_texts:
        zs_bios.append(clf(x, candidate_labels=occ_labels))

    print("Zero-shot sentiment (examples):")
    sent_examples = []
    for s in zs_sent:
        sent_examples.append(
            (s["sequence"][:40] + "...", s["labels"][0], round(s["scores"][0], 3))
        )
    print(sent_examples)

    print("Zero-shot bios (examples):")
    bios_examples = []
    for s in zs_bios:
        bios_examples.append(
            (s["sequence"][:40] + "...", s["labels"][0], round(s["scores"][0], 3))
        )
    print(bios_examples)



if __name__ == "__main__":
    main()
