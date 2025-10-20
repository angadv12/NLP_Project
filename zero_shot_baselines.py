import pandas as pd
from datasets import load_dataset
from transformers import pipeline


def main():
    sst = load_dataset("glue", "sst2")
    sst_texts = []
    for i in [0, 10, 20, 30, 40]:
      example = sst["validation"][i]
      sentence = example["sentence"]
      sst_texts.append(sentence)

    bios = pd.read_csv("data/bios_subset.csv")
    occ_labels = sorted(bios["occupation"].unique())
    bios_texts = bios["text"].tolist()[:8]

    clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
    zs_sent = []
    for x in sst_texts:
      result = clf(x, candidate_labels=["positive", "negative"])
      zs_sent.append(result)
    zs_bios = []
    for x in bios_texts:
      result = clf(x, candidate_labels=occ_labels)
      zs_bios.append(result)

    print("Zero-shot sentiment (examples):")
    sent_examples = []
    for s in zs_sent:
      sequence = s["sequence"][:40] + "..."
      label = s["labels"][0]
      score = round(s["scores"][0], 3)
      sent_examples.append((sequence, label, score))
    print(sent_examples)

    print("Zero-shot bios (examples):")
    bios_examples = []
    for s in zs_bios:
      sequence = s["sequence"][:40] + "..."
      label = s["labels"][0]
      score = round(s["scores"][0], 3)
      bios_examples.append((sequence, label, score))
    print(bios_examples)


if __name__ == "__main__":
    main()
