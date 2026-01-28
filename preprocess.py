"""
Preprocessing script for ED (Emotion Detection) dataset.
Downloads and processes the emotion classification dataset from GitHub.
"""
import html
import os
import re

import fire
import pandas as pd
import requests
from datasets import DatasetDict, Dataset
from tqdm import tqdm

# ED dataset: 32 emotion labels with index mapping
ED_LABEL2IDX = {
    'sad': 0, 'trusting': 1, 'terrified': 2, 'caring': 3, 'disappointed': 4,
    'faithful': 5, 'joyful': 6, 'jealous': 7, 'disgusted': 8, 'surprised': 9,
    'ashamed': 10, 'afraid': 11, 'impressed': 12, 'sentimental': 13,
    'devastated': 14, 'excited': 15, 'anticipating': 16, 'annoyed': 17, 'anxious': 18,
    'furious': 19, 'content': 20, 'lonely': 21, 'angry': 22, 'confident': 23,
    'apprehensive': 24, 'guilty': 25, 'embarrassed': 26, 'grateful': 27,
    'hopeful': 28, 'proud': 29, 'prepared': 30, 'nostalgic': 31
}
ED_IDX2LABEL = {v: k for k, v in ED_LABEL2IDX.items()}
ED_EMOTION_LABELS = list(ED_LABEL2IDX.keys())

# ED_easy_4 dataset: 4 emotion labels with index mapping
ED_EASY_4_LABELS = {
    0: "sad",
    1: "joyful",
    2: "angry",
    3: "afraid",
}

# ED_hard_a labels are derived from label_tree.tsv order (excluding roots)
ED_HARD_A_DATASET = "ED_hard_a"

# Identity mapping - ED dataset uses its native emotion labels directly
ED_LABEL_TREE_MAPPING = {label: label for label in ED_EMOTION_LABELS}

# GitHub raw content URLs for ED dataset
ED_GITHUB_BASE = "https://raw.githubusercontent.com/dinobby/HypEmo/main/data/ED"
ED_FILES = {
    "train": f"{ED_GITHUB_BASE}/train.csv",
    "valid": f"{ED_GITHUB_BASE}/valid.csv",
    "test": f"{ED_GITHUB_BASE}/test.csv",
    "label_tree": f"{ED_GITHUB_BASE}/label_tree.tsv"
}


def download_ed_dataset(data_root_dir="data"):
    """Download ED emotion dataset from GitHub."""
    ed_dir = os.path.join(data_root_dir, "ED")
    os.makedirs(ed_dir, exist_ok=True)
    
    print(f"Downloading ED emotion dataset from GitHub to {ed_dir}")
    
    for file_name, url in ED_FILES.items():
        local_path = os.path.join(ed_dir, f"{file_name}.csv" if file_name != "label_tree" else "label_tree.tsv")
        
        if os.path.exists(local_path):
            print(f"{os.path.basename(local_path)} already exists, skipping download")
            continue
        
        print(f"Downloading {file_name}...")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                f.write(response.content)
            
            file_size = len(response.content) / 1024 / 1024
            print(f"Downloaded {os.path.basename(local_path)} ({file_size:.2f} MB)")
        except Exception as e:
            print(f"Error downloading {file_name}: {e}")
            raise
    
    print("Download complete!")
    return ed_dir


def clean_text(raw_text: str) -> str:
    """Clean and normalize text."""
    if not raw_text or pd.isna(raw_text):
        return ""
    text = html.unescape(str(raw_text))
    text = text.strip()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[\n\t]", " ", text)
    text = re.sub(r" +", " ", text)
    text = re.sub(r"[^\x00-\x7F]", " ", text)
    return text.strip()


def preprocess_emotion_dataset(data_root_dir="data", output_name="ED_processed", download=True, dataset_name="ED"):
    """
    Preprocess emotion classification dataset from ED format.
    
    Args:
        data_root_dir: Root directory for data storage
        output_name: Name for the output processed dataset
        download: Whether to download the dataset from GitHub
    
    Downloads from: https://github.com/dinobby/HypEmo/tree/main/data/ED (only for dataset_name="ED")
    
    Expected input structure:
        data/ED/... for dataset_name="ED" (auto-downloaded)
        data/ED_easy_4/... for dataset_name="ED_easy_4" (local files)
        Each dataset dir contains:
        - train.csv, valid.csv, test.csv with columns: text, aug_text, label
        - label_tree.tsv (optional for ED_easy_4; not used for label mapping)
    
    Output structure:
        data/{output_name}/ - HuggingFace datasets format with train/valid/test/emotion_info/item_info splits
    """
    dataset_dir_name = dataset_name
    if dataset_dir_name == "ED" and download:
        ed_dir = download_ed_dataset(data_root_dir)
    else:
        ed_dir = os.path.join(data_root_dir, dataset_dir_name)
        if not os.path.exists(ed_dir):
            raise ValueError(f"Emotion dataset directory not found: {ed_dir}.")
    
    output_dir = os.path.join(data_root_dir, output_name)
    
    print(f"\nProcessing emotion dataset from {ed_dir}")
    
    # Load label tree
    label_tree_path = os.path.join(ed_dir, "label_tree.tsv")
    label_to_name = {}
    emotion_labels = None
    is_easy_4 = dataset_dir_name.lower() == "ed_easy_4"
    is_hard_a = dataset_dir_name.lower() == ED_HARD_A_DATASET.lower()
    if is_easy_4:
        label_to_name = ED_EASY_4_LABELS.copy()
        emotion_labels = [ED_EASY_4_LABELS[i] for i in sorted(ED_EASY_4_LABELS.keys())]

    if os.path.exists(label_tree_path):
        label_tree_df = pd.read_csv(label_tree_path, sep="\t", header=None, names=["emotion", "parent"])
        unique_emotions = label_tree_df["emotion"].unique()
        if is_hard_a:
            ordered = []
            for emotion in label_tree_df["emotion"].tolist():
                e = str(emotion).strip().lower()
                if e in {"negative", "positive", "root"}:
                    continue
                if e not in ordered:
                    ordered.append(e)
            label_to_name = {i: e for i, e in enumerate(ordered)}
            emotion_labels = ordered
            print(f"Loaded {len(ordered)} emotions from label tree (ED_hard_a)")
        elif not is_easy_4:
            for i, emotion in enumerate(unique_emotions):
                label_to_name[i] = emotion
            print(f"Loaded {len(unique_emotions)} emotions from label tree")
    else:
        print("Warning: label_tree.tsv not found")
    
    if emotion_labels is None:
        emotion_labels = ED_EMOTION_LABELS

    # Load and process train/valid/test datasets
    datasets = {}
    for split in ["train", "valid", "test"]:
        csv_path = os.path.join(ed_dir, f"{split}.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: {split}.csv not found, skipping")
            continue
        
        df = pd.read_csv(csv_path)
        print(f"Loaded {split}: {len(df)} samples")
        
        # Process the dataset
        processed_data = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split}"):
            text = row["text"] if pd.notna(row["text"]) else ""
            aug_text = row["aug_text"] if "aug_text" in row and pd.notna(row["aug_text"]) else text
            label_id = int(row["label"])
            
            # Map label to emotion name
            if label_id in label_to_name:
                emotion_raw = label_to_name[label_id]
                if is_easy_4 or is_hard_a:
                    emotion = emotion_raw.lower()
                else:
                    emotion = ED_LABEL_TREE_MAPPING.get(emotion_raw.lower(), emotion_raw.lower())
                    if emotion not in [e.lower() for e in ED_EMOTION_LABELS]:
                        if idx < 5:
                            print(f"Warning: Unknown emotion '{emotion_raw}' (mapped to '{emotion}'), using 'neutral'")
                        emotion = "neutral"
            else:
                if idx < 5:
                    print(f"Warning: Label {label_id} not found in label tree, using 'neutral'")
                emotion = "neutral"
            
            processed_data.append({
                "text": clean_text(text),
                "aug_text": clean_text(aug_text) if aug_text != text else clean_text(text),
                "label": label_id,
                "seq_labels": label_id,
                "emotion": emotion,
                "review_id": f"{split}_{idx}",
                "interaction_id": f"{split}_{idx}",
            })
        
        datasets[split] = Dataset.from_pandas(pd.DataFrame(processed_data))
    
    # Create emotion label info dataset
    emotion_info = []
    positive_emotions = ["joy", "excitement", "love", "gratitude", "admiration", "approval", 
                        "caring", "optimism", "pride", "relief", "amusement", "awe"]
    negative_emotions = ["anger", "annoyance", "disappointment", "disapproval", "disgust", 
                        "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness", 
                        "contempt", "envy", "guilt", "shame"]
    
    for i, emotion in enumerate(emotion_labels):
        category = "positive" if emotion in positive_emotions else \
                  ("negative" if emotion in negative_emotions else "neutral")
        emotion_info.append({
            "emotion_id": i,
            "emotion_name": emotion,
            "emotion_category": category,
        })

    item_info = []
    for i, emotion in enumerate(emotion_labels):
        item_info.append({
            "item_id": i,
            "title": emotion,
        })
    
    # Create DatasetDict
    dataset_dict = DatasetDict({
        "train": datasets.get("train"),
        "valid": datasets.get("valid"),
        "test": datasets.get("test"),
        "emotion_info": Dataset.from_pandas(pd.DataFrame(emotion_info)),
        "item_info": Dataset.from_pandas(pd.DataFrame(item_info)),
    })
    
    # Remove None entries
    dataset_dict = DatasetDict({k: v for k, v in dataset_dict.items() if v is not None})
    
    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    dataset_dict.save_to_disk(output_dir)
    
    print(f"\nDataset saved to {output_dir}")
    print(f"Train: {len(dataset_dict['train'])}, "
          f"Valid: {len(dataset_dict['valid'])}, "
          f"Test: {len(dataset_dict['test'])}, "
          f"Emotions: {len(dataset_dict['emotion_info'])}")
    
    return output_dir


def main(
    data_root_dir: str = "data",
    output_name: str = "ED_processed",
    download: bool = True,
    dataset_name: str = "ED",
):
    """
    Main preprocessing function for ED emotion classification dataset.
    
    Args:
        data_root_dir: Root directory for data storage (default: "data")
        output_name: Name for the output processed dataset (default: "ED_processed")
        download: Whether to download the dataset from GitHub (default: True)
    
    Example usage:
        python preprocess.py
        python preprocess.py --data_root_dir=/path/to/data --output_name=emotion_data
        python preprocess.py --download=False  # Use existing local files
        python preprocess.py --dataset_name=ED_easy_4 --download=False --output_name=ED_easy_4_processed
    """
    print("=" * 80)
    print("ED Emotion Dataset Preprocessing")
    print("Source: https://github.com/dinobby/HypEmo/tree/main/data/ED")
    print("=" * 80)
    print(f"Data directory: {data_root_dir}")
    print(f"Output name: {output_name}")
    print(f"Download from GitHub: {download}")
    print("=" * 80)
    
    preprocess_emotion_dataset(
        data_root_dir=data_root_dir,
        output_name=output_name,
        download=download,
        dataset_name=dataset_name,
    )
    
    print("\nPreprocessing complete!")


if __name__ == "__main__":
    fire.Fire(main)
