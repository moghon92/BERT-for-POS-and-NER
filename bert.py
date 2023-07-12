import torch

torch.manual_seed(10)
import torch.nn as nn
from transformers import BertModel


class BERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BERTClassifier, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(768, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, inputs, mask):
        output = self.model(inputs, mask)

        x = output.pooler_output
        x = self.dropout(x)
        x = self.linear(x)

        return x
