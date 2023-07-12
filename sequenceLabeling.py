import torch

torch.manual_seed(10)
import torch.nn as nn
from transformers import BertModel


class SequenceLabeling(nn.Module):
    def __init__(self, num_classes):
        """
        Initializing the following modules:
            1. Bert Model using the pretrained 'bert-base-uncased' model,
            2. Dropout module.
            3. Linear layer. In dimension should be 768.

        Args:
        num_classes: Number of classes (labels).

        """
        super(SequenceLabeling, self).__init__()

        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(768, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, inputs, mask, token_type_ids):
        """
        Implementing the forward function to feed the input through the bert model with inputs, mask and token type ids.
        The output of bert layer model is then fed to dropout, linear and relu.

        Args:
            inputs: Input data.
            mask: attention_mask
            token_type_ids: token type ids

        Returns:
          output: Logits of each label.
        """
        output = self.model(inputs, mask, token_type_ids)

        x = output.last_hidden_state
        x = self.dropout(x)
        x = self.linear(x).relu()

        return x