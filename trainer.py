import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from preprocess import convert_data_to_features


def collate_fn(batch):
    max_len = max([len(x['input_ids']) for x in batch])

    input_ids = [x['input_ids'] + ([0] * (max_len - len(x['input_ids']))) for x in batch]
    attention_mask = [([1.0] * len(x['input_ids'])) + ([0.0] * (max_len - len(x['input_ids']))) for x in batch]

    labels = [x['labels'] for x in batch]
    entity_positions = [x['entity_positions'] for x in batch]
    head_tail_pairs = [x['head_tail_pairs'] for x in batch]

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.float)

    output = (input_ids, attention_mask, entity_positions, head_tail_pairs, labels)

    return output


class Trainer():
    def __init__(self, args, config, model, tokenizer):
        self.args = args
        self.config = config

        self.train_data = self.load_data(filepath=args.train_file)
        self.dev_data = self.load_data(filepath=args.dev_file)

        if args.debug:
            self.train_data = self.train_data[:100]
            self.dev_data = self.dev_data[:100]

        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
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

        epoch_pbar = trange(self.num_epochs, desc="Epoch", total=self.num_epochs)
        for epoch in epoch_pbar:
            train_pbar = tqdm(iterable=train_dataloader, desc="Training", total=len(train_dataloader))
            for step, batch in enumerate(train_pbar):
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'entity_positions': batch[2],
                          'head_tail_pairs': batch[3],
                          'labels': batch[4]}

                outputs = self.model(**inputs)

    def evaluate(self,):
        pass
