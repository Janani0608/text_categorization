import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd 
import re

class MedicalTextDataset(Dataset):

    def __init__(self, data_frame, tokenizer, label_encoder, max_length = 512):
        self.data_frame = data_frame
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.max_length = max_length

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        
        row = self.data_frame.iloc[idx]

        # Convert to string and handle missing values
        description = str(row['description']) if pd.notna(row['description']) else ""
        transcription = str(row['transcription']) if pd.notna(row['transcription']) else ""

        text = description + " " + transcription  # Concatenate safely
        encoding = self.tokenizer(
            text,
            truncation = True,
            padding = 'max_length',
            max_length = self.max_length,
            return_tensors = 'pt'
        )

        label = self.label_encoder.transform([row['medical_specialty']])[0]

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(filepath):
    return pd.read_csv(filepath)
