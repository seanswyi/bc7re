import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers.optimization import get_linear_schedule_with_warmup
import wandb

from official_evaluation.compute_metrics import compute_metrics
from official_evaluation.utils import format_relations, get_chemical_gene_combinations, load_entities_dict, preprocess_data
from preprocess import convert_data_to_features


with open(file='/hdd1/seokwon/data/BC7DP/relation2id.json') as f:
    relation2id = json.load(fp=f)


id2relation = {id_: relation for relation, id_ in relation2id.items()}
relation_names = [x for x in relation2id if x != 'Na']
relname2tag = {name: idx + 1 for idx, name in enumerate(relation_names)}
num_true_relations = len(relation_names)


def convert_to_df(data, mode='original'):
    temp_data = []
    for sample in data:
        if mode == 'original':
            pmid = sample['pmid']

            for label in sample['relations']:
                relation_name = label['relation']
                head_id = f"Arg:{label['head_id']}"
                tail_id = f"Arg:{label['tail_id']}"

                temp_data.append([pmid, relation_name, head_id, tail_id])
        elif mode == 'preds':
            pmid = sample['pmid']
            relation_name = sample['relation']
            head_id = f"Arg:{sample['head_id']}"
            tail_id = f"Arg:{sample['tail_id']}"

            temp_data.append([pmid, relation_name, head_id, tail_id])

    data_df = pd.DataFrame(temp_data, columns=['pmid', 'rel_type', 'arg1', 'arg2'])

    return data_df


def collate_fn(batch):
    max_len = max([len(x['input_ids']) for x in batch])

    input_ids = [x['input_ids'] + ([0] * (max_len - len(x['input_ids']))) for x in batch]
    attention_mask = [([1.0] * len(x['input_ids'])) + ([0.0] * (max_len - len(x['input_ids']))) for x in batch]

    labels = [x['labels'] for x in batch]
    entity_positions = [x['entity_positions'] for x in batch]
    head_tail_pairs = [x['head_tail_pairs'] for x in batch]

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.float)

    entity_set = [x['entity_set'] for x in batch]

    output = (input_ids, attention_mask, entity_positions, entity_set, head_tail_pairs, labels)

    return output


class Trainer():
    def __init__(self, args, config, model, optimizer, tokenizer):
        self.args = args
        self.config = config

        self.train_data = self.load_data(filepath=args.train_file)
        self.dev_data = self.load_data(filepath=args.dev_file)
        self.dev_df = convert_to_df(data=self.dev_data)

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
        self.entity_marker = args.entity_marker
        self.evaluation_step = args.evaluation_step
        self.negative_ratio = args.negative_ratio
        self.num_epochs = args.num_epochs
        self.learning_rate = args.learning_rate
        self.use_at_loss = args.use_at_loss
        self.warmup_ratio = args.warmup_ratio

        self.entity_file = args.entity_file

        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer

        self.timestamp = args.timestamp
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
        for _ in epoch_pbar:
            self.model.zero_grad()

            train_pbar = tqdm(iterable=train_dataloader, desc="Training", total=len(train_dataloader))
            for batch in train_pbar:
                self.model.train()

                inputs = {'input_ids': batch[0].to('cuda'),
                          'attention_mask': batch[1].to('cuda'),
                          'entity_positions': batch[2],
                          'entity_set': batch[3],
                          'head_tail_pairs': batch[4],
                          'labels': batch[5]}

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

                        if not self.args.dont_save:
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
        answers_df = convert_to_df(data=answers, mode='preds')

        pmid2chemicals_and_genes, _, chemicals = load_entities_dict(path=self.entity_file)
        pmid2combinations, num_combinations = get_chemical_gene_combinations(input_dict=pmid2chemicals_and_genes)
        pmids = set(map(lambda x: str(x.strip()), [x['pmid'] for x in self.dev_data]))

        true_valid, true_relation_list = preprocess_data(df=self.dev_df, chemicals=chemicals, rel_types=relation_names, is_gs=True)
        answers_valid, answers_relation_list = preprocess_data(df=answers_df, chemicals=chemicals, rel_types=relation_names, gs_files=pmids)

        y_true, y_pred = format_relations(gs_valid=true_valid,
                                          preds_valid=answers_valid,
                                          pmid2combinations=pmid2combinations,
                                          num_combinations=num_combinations,
                                          num_relations=num_true_relations,
                                          relname2tag=relname2tag)

        result_dict = compute_metrics(y_true=y_true,
                                      y_pred=y_pred,
                                      relname2tag=relname2tag,
                                      gs_rel_list=true_relation_list,
                                      preds_rel_list=answers_relation_list)

        return result_dict, all_predictions, averaged_eval_loss

    def convert_to_evaluation_features(self, predictions, features):
        head_idxs = []
        tail_idxs = []
        pmids = []

        all_results = []

        for feature in features:
            head_tail_pairs = feature['head_tail_pairs']
            head_idxs += [('T' + str(pair[0] + 1)) for pair in head_tail_pairs]
            tail_idxs += [('T' + str(pair[1] + 1)) for pair in head_tail_pairs]
            pmids += [feature['pmid'] for pair in head_tail_pairs]

        results = []
        for idx, prediction in enumerate(predictions):
            if self.use_at_loss:
                prediction = np.nonzero(prediction)[0].tolist()

                for pred in prediction:
                    result_dict = {'pmid': pmids[idx],
                                   'head_id': head_idxs[idx],
                                   'tail_id': tail_idxs[idx],
                                   'relation': id2relation[pred]}

                    if pred != 0:
                        results.append(result_dict)

                    all_results.append(result_dict)
            else:
                prediction = int(predictions[idx])

                result_dict = {'pmid': pmids[idx],
                               'head_id': head_idxs[idx],
                               'tail_id': tail_idxs[idx],
                               'relation': id2relation[prediction]}

                if prediction != 0:
                    results.append(result_dict)

                all_results.append(result_dict)

        return results, all_results
