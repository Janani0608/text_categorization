import torch
import torch.nn as nn
from transformers import BertModel
from transformers import DistilBertModel

class TextClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super(TextClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        return self.fc(cls_output)
