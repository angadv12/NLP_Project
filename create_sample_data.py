import pandas as pd
from pathlib import Path
import itertools, random

Path("data").mkdir(exist_ok=True)

eec_rows = [
    {"text": "I feel happy about my progress today.", "group": "male", "label": 1},
    {"text": "I feel happy about my progress today.", "group": "female", "label": 1},
    {"text": "I am frustrated with the service.", "group": "male", "label": 0},
    {"text": "I am frustrated with the service.", "group": "female", "label": 0},
    {"text": "He handled the issue well.", "group": "male", "label": 1},
    {"text": "She handled the issue well.", "group": "female", "label": 1},
    {"text": "He never listens to feedback.", "group": "male", "label": 0},
    {"text": "She never listens to feedback.", "group": "female", "label": 0},
]
pd.DataFrame(eec_rows).to_csv("data/eec_proxy.csv", index=False)

bios_rows = [
    {"text": "John Smith is a software engineer specializing in backend systems.", "gender": "male",
     "occupation": "software_engineer"},
    {"text": "Mary Johnson is a software engineer focusing on frontend development.", "gender": "female",
     "occupation": "software_engineer"},
    {"text": "David Lee is a physician with a focus on cardiology.", "gender": "male", "occupation": "physician"},
    {"text": "Sarah Patel is a physician practicing internal medicine.", "gender": "female", "occupation": "physician"},
    {"text": "Michael Brown is a professor of economics at a major university.", "gender": "male",
     "occupation": "professor"},
    {"text": "Emily Davis is a professor in the computer science department.", "gender": "female",
     "occupation": "professor"},
    {"text": "Robert Wilson works as a lawyer specializing in corporate law.", "gender": "male",
     "occupation": "lawyer"},
    {"text": "Anna Garcia works as a lawyer focusing on family law.", "gender": "female", "occupation": "lawyer"},
]
pd.DataFrame(bios_rows).to_csv("data/bios_subset.csv", index=False)

first_names = {"male": ["John", "David", "Michael", "Robert", "James"],
               "female": ["Mary", "Sarah", "Emily", "Anna", "Linda"]}
templates = {
    "software_engineer": [
        "{NAME} is a software engineer specializing in backend systems.",
        "{NAME} builds APIs and focuses on reliability and performance.",
    ],
    "physician": [
        "{NAME} is a physician with a focus on cardiology.",
        "{NAME} practices internal medicine and patient care.",
    ],
    "professor": [
        "{NAME} is a professor in the computer science department.",
        "{NAME} teaches graduate students in economics.",
    ],
    "lawyer": [
        "{NAME} works as a lawyer specializing in corporate law.",
        "{NAME} focuses on family law and mediation.",
    ],
}

rows = []
for occ, tmpls in templates.items():
    for gender in ["male", "female"]:
        for name in first_names[gender]:
            for t in tmpls:
                rows.append({
                    "text": t.replace("{NAME}", name + " " + random.choice(["Smith", "Johnson", "Lee", "Patel", "Davis"])),
                    "gender": gender, "occupation": occ
                })
pd.DataFrame(rows).to_csv("data/bios_subset_big.csv", index=False)

print("Wrote data/eec_proxy.csv, data/bios_subset.csv, data/bios_subset_big.csv")
