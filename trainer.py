import json

import torch
from torch.utils.data import DataLoader

from preprocess import convert_data_to_features


def collate_fn(batch):
    max_len = max([len(x['input_ids']) for x in batch])

    input_ids = [x['input_ids'] + ([0] * (max_len - len(x['input_ids']))) for x in batch]
    attention_mask = [([1.0] * len(x['input_ids'])) + ([0.0] * (max_len - len(x['input_ids']))) for x in batch]

    labels = [x['labels'] for x in batch]
    entity_positions = [x['entity_positions'] for x in batch]

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.float)

    output = (input_ids, attention_mask, labels, entity_positions)

    return output


class Trainer():
    def __init__(self, args, config, model, tokenizer):
        self.train_data = self.load_data(filepath=args.train_file)
        self.dev_data = self.load_data(filepath=args.dev_file)

        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate

        self.entity_marker = args.entity_marker

        self.model = model
        self.tokenizer = tokenizer

    def load_data(self, filepath):
        with open(file=filepath) as f:
            data = json.load(fp=f)

        return data

    def train(self,):
        train_features = convert_data_to_features(data=self.train_data, tokenizer=self.tokenizer, entity_marker=self.entity_marker)
        train_dataloader = DataLoader(train_features, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=False)
        import pdb; pdb.set_trace()

    def evaluate(self,):
        pass
