import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset, Features, Value
from transformers import AutoTokenizer

# Function to load the dataset
def load_scientific_papers(split):
    features = Features({
        'article': Value('string'),
        'abstract': Value('string')
    })

    if split == "train":
        return load_dataset(
            "armanc/scientific_papers",
            name="arxiv",
            trust_remote_code=True,
            split=f"train[:1000]",
            features=features
        )
    elif split == "validation":
        return load_dataset(
            "armanc/scientific_papers",
            name="arxiv",
            trust_remote_code=True,
            split=f"validation[:100]",
            features=features
        )
    else:
        raise ValueError(f"Unknown split: {split}")
