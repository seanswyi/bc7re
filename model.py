import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class DrugProtREModel(nn.Module):
    def __init__(self, args, model):
        super().__init__()

        if args.backbone == 'biobert':
            self.backbone_model = AutoModel.from_pretrained('dmis-lab/biobert-v1.1')
        elif args.backbone == 'biolm':
            raise NotImplementedError

    def encode(self, input_ids):
        pass

    def forward(self, x):
        return x
