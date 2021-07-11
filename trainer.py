import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers.optimization import get_linear_schedule_with_warmup
import wandb

from preprocess import convert_data_to_features


with open(file='/hdd1/seokwon/data/BC7DP/relation2id.json') as f:
    relation2id = json.load(fp=f)
    id2relation = {id_: relation for relation, id_ in relation2id.items()}


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
    def __init__(self, args, config, model, optimizer, tokenizer):
        self.args = args
        self.config = config

        self.train_data = self.load_data(filepath=args.train_file)
        self.dev_data = self.load_data(filepath=args.dev_file)

        if args.debug:
            self.train_data = self.train_data[:100]
            self.dev_data = self.dev_data[:100]

        self.train_features = convert_data_to_features(data=self.train_data,
                                                       tokenizer=tokenizer,
                                                       negative_ratio=args.negative_ratio,
                                                       entity_marker=args.entity_marker)
        self.dev_features = convert_data_to_features(data=self.dev_data,
                                                     tokenizer=tokenizer,
                                                     negative_ratio=args.negative_ratio,
                                                     entity_marker=args.entity_marker,
                                                     mode='dev')

        self.batch_size = args.batch_size
        self.evaluation_step = args.evaluation_step
        self.negative_ratio = args.negative_ratio
        self.num_epochs = args.num_epochs
        self.learning_rate = args.learning_rate
        self.warmup_ratio = args.warmup_ratio

        self.entity_marker = args.entity_marker

        self.use_at_loss = args.use_at_loss

        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer

        self.wandb_name = args.wandb_name

        self.checkpoint_save_dir = os.path.join(args.checkpoint_save_dir, args.wandb_name)
        self.pred_save_dir = os.path.join(args.pred_save_dir, args.wandb_name)

        if not os.path.exists(self.checkpoint_save_dir):
            os.makedirs(self.checkpoint_save_dir, exist_ok=True)

        if not os.path.exists(self.pred_save_dir):
            os.makedirs(self.pred_save_dir, exist_ok=True)

    def load_data(self, filepath):
        with open(file=filepath) as f:
            data = json.load(fp=f)

        return data

    def train(self,):
        train_dataloader = DataLoader(self.train_features, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=False)

        total_steps = int(len(train_dataloader) * self.num_epochs)
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)

        best_score = 0
        num_steps = 0
        epoch_pbar = trange(self.num_epochs, desc="Epoch", total=self.num_epochs)
        for epoch in epoch_pbar:
            self.model.zero_grad()

            train_pbar = tqdm(iterable=train_dataloader, desc="Training", total=len(train_dataloader))
            for step, batch in enumerate(train_pbar):
                self.model.train()

                inputs = {'input_ids': batch[0].to('cuda'),
                          'attention_mask': batch[1].to('cuda'),
                          'entity_positions': batch[2],
                          'head_tail_pairs': batch[3],
                          'labels': batch[4]}

                outputs = self.model(**inputs)

                loss = outputs[-1]

                loss.backward()
                self.optimizer.step()
                scheduler.step()
                self.model.zero_grad()

                wandb.log({'param_group1_lr': scheduler.get_last_lr()[0], 'param_group2_lr': scheduler.get_last_lr()[1]}, step=num_steps)
                wandb.log({'loss': loss.item()}, step=num_steps)

                if num_steps % self.evaluation_step == 0:
                    results, all_predictions, averaged_eval_loss = self.evaluate()
                    wandb.log(results, step=num_steps)
                    wandb.log({'eval_loss': averaged_eval_loss}, step=num_steps)

                    print(f"Step {num_steps} results: {results}")

                    if results['f1'] >= best_score:
                        best_score = results['f1']

                        checkpoint_file = f'{self.wandb_name}.pt'
                        checkpoint_filename = os.path.join(self.checkpoint_save_dir, checkpoint_file)
                        torch.save(self.model.state_dict(), checkpoint_filename)

                        pred_file = f'{self.wandb_name}_preds_step-{num_steps}.json'
                        pred_filename = os.path.join(self.pred_save_dir, pred_file)
                        with open(file=pred_filename, mode='w') as f:
                            json.dump(obj=all_predictions, fp=f, indent=2)

                num_steps += 1

    def evaluate(self, mode='dev'):
        dev_dataloader = DataLoader(self.dev_features, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)

        with open(file='/hdd1/seokwon/BC7/dev_samples_with_negative.json', mode='w') as f:
            json.dump(obj=self.dev_features, fp=f, indent=2)

        total_eval_loss = 0
        preds = []
        dev_pbar = tqdm(iterable=dev_dataloader, desc=f"Evaluating {mode}", total=len(dev_dataloader))
        for batch in dev_pbar:
            self.model.eval()

            inputs = {'input_ids': batch[0].to('cuda'),
                      'attention_mask': batch[1].to('cuda'),
                      'entity_positions': batch[2],
                      'head_tail_pairs': batch[3]}

            if mode == 'dev':
                inputs['labels'] = batch[4]

            with torch.no_grad():
                output = self.model(**inputs)
                pred = output[0]
                pred = pred.cpu().numpy()
                preds.append(pred)

                loss = output[-1]
                total_eval_loss += loss.detach().cpu().item()

        averaged_eval_loss = total_eval_loss / len(dev_dataloader)
        predictions = np.concatenate(preds, axis=0).astype(np.float32)
        answers, all_predictions = self.convert_to_evaluation_features(predictions, self.dev_features)

        if answers == []:
            precision = 0
            recall = 0
            f1 = 0
        else:
            tp = 0

            true_converted = []
            for document in self.dev_data:
                pmid = document['doc_id']
                relations = document['relations']

                for relation in relations:
                    template = {'pmid': pmid, 'head_id': relation['head_id'], 'tail_id': relation['tail_id'], 'relation': relation['relation']}
                    true_converted.append(template)

            positive_predictions = [x for x in answers if x['relation'] != 'Na']
            tp_plus_fp = len(positive_predictions)
            tp_plus_fn = len(true_converted)

            for prediction in positive_predictions:
                if prediction in true_converted:
                    tp += 1

            if tp_plus_fp == 0:
                precision = 0
            else:
                precision = tp / tp_plus_fp

            if tp_plus_fn == 0:
                recall = 0
            else:
                recall = tp / tp_plus_fn

            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)

        return {'precision': precision, 'recall': recall, 'f1': f1}, all_predictions, averaged_eval_loss

    def convert_to_evaluation_features(self, predictions, features):
        head_idxs = []
        tail_idxs = []
        doc_ids = []

        all_results = []

        for feature in features:
            head_tail_pairs = feature['head_tail_pairs']
            head_idxs += [('T' + str(pair[0] + 1)) for pair in head_tail_pairs]
            tail_idxs += [('T' + str(pair[1] + 1)) for pair in head_tail_pairs]
            doc_ids += [feature['doc_id'] for pair in head_tail_pairs]

        results = []
        for idx, prediction in enumerate(predictions):
            if self.use_at_loss:
                prediction = np.nonzero(prediction)[0].tolist()

                for pred in prediction:
                    result_dict = {'pmid': doc_ids[idx],
                                   'head_id': head_idxs[idx],
                                   'tail_id': tail_idxs[idx],
                                   'relation': id2relation[pred]}

                    if pred != 0:
                        results.append(result_dict)

                    all_results.append(result_dict)
            else:
                prediction = int(predictions[idx])

                result_dict = {'pmid': doc_ids[idx],
                               'head_id': head_idxs[idx],
                               'tail_id': tail_idxs[idx],
                               'relation': id2relation[prediction]}

                if prediction != 0:
                    results.append(result_dict)

                all_results.append(result_dict)

        return results, all_results
