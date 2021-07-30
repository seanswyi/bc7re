import argparse
from datetime import datetime
import logging
import os
import time

import torch.optim as optim
from transformers import AutoConfig, AutoModel, AutoTokenizer
import wandb

from model import DrugProtREModel
from trainer import Trainer


logger = logging.getLogger()


def main(args):
    args.train_file = os.path.join(args.data_dir, 'training', args.train_file)
    args.dev_file = os.path.join(args.data_dir, 'development', args.dev_file)
    args.entity_file = os.path.join(args.data_dir, 'development', args.entity_file)

    logger.info('Arguments:')
    for name, value in sorted([(x, y) for x, y in vars(args).items()], key=lambda x: x[0]):
        logger.info(f'\t{name}: {value}')

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    backbone_model = AutoModel.from_pretrained(args.model_name_or_path)

    if args.entity_marker == 'typed':
        special_tokens_dict = {'additional_special_tokens': ['[CHEMICAL]', '[GENE]']}
        _ = tokenizer.add_special_tokens(special_tokens_dict)
        backbone_model.resize_token_embeddings(len(tokenizer))

    model = DrugProtREModel(args=args, config=config, backbone_model=backbone_model, tokenizer=tokenizer)
    model = model.to('cuda')

    new_layer = ['extractor', 'classifier']
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], 'lr': args.classifier_learning_rate},
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    trainer = Trainer(args=args, config=config, model=model, optimizer=optimizer, tokenizer=tokenizer)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--adam_epsilon', default=1e-6, type=float)
    parser.add_argument('--adaptive_thresholding_k', default=1, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--bilinear_block_size', default=64, type=int)
    parser.add_argument('--checkpoint_save_dir', default='/hdd1/seokwon/BC7/checkpoints', type=str)
    parser.add_argument('--classifier_learning_rate', default=1e-5, type=float)
    parser.add_argument('--classification_type', default='entity_marker', choices=['both', 'cls', 'entity_marker'])
    parser.add_argument('--classifier_type', default='bilinear', choices=['linear', 'bilinear'])
    parser.add_argument('--data_dir', default='/hdd1/seokwon/data/BC7DP/drugprot-gs-training-development', type=str)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--dev_file', default='dev.json', type=str)
    parser.add_argument('--dont_save', action='store_true', default=False)
    parser.add_argument('--entity_marker', default='asterisk', type=str, choices=['asterisk', 'typed'])
    parser.add_argument('--entity_file', default='drugprot_development_entities.tsv')
    parser.add_argument('--evaluation_step', default=250, type=int)
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--log_file_dir', default='/hdd1/seokwon/BC7/logs', type=str)
    parser.add_argument('--max_seq_len', default=512, type=str)
    parser.add_argument('--model_name_or_path', default='dmis-lab/biobert-v1.1', type=str, \
        choices=['dmis-lab/biobert-v1.1', '/hdd1/seokwon/BC7/checkpoints/RoBERTa-base-PM-M3-Voc-distill/'])
    parser.add_argument('--negative_ratio', default=0.5, type=float)
    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--num_labels', default=14, type=int)
    parser.add_argument('--pred_save_dir', default='/hdd1/seokwon/BC7/predictions', type=str)
    parser.add_argument('--setting', default='document', type=str, choices=['document', 'sentence'])
    parser.add_argument('--train_file', default='train.json', type=str)
    parser.add_argument('--use_at_loss', action='store_true', default=False)
    parser.add_argument('--use_attention', action='store_true', default=False)
    parser.add_argument('--warmup_ratio', default=0.05, type=float)

    args = parser.parse_args()

    time_right_now = time.time()
    timestamp = datetime.fromtimestamp(timestamp=time_right_now).strftime(format='%m-%d-%Y-%H%M')
    args.timestamp = timestamp

    if 'biobert' in args.model_name_or_path:
        model_name = 'biobert'
    elif 'roberta' in args.model_name_or_path.lower():
        model_name = 'biolm'

    wandb_name = f'{model_name}_{args.setting}_{args.classification_type}_{args.classifier_type}_{args.learning_rate}_{args.classifier_learning_rate}_{args.negative_ratio}_{args.timestamp}'

    if args.use_at_loss:
        wandb_name = 'AT_' + wandb_name

    if args.use_attention:
        wandb_name = wandb_name.replace('AT_', 'ATLOP_')

    args.wandb_name = wandb_name
    wandb.init(project='BC7DP', name=wandb_name)

    if not os.path.exists(args.log_file_dir):
        os.makedirs(args.log_file_dir, exist_ok=True)

    args.log_filename = os.path.join(args.log_file_dir, f'{wandb_name}.log')

    log_msg_format = '[%(asctime)s - %(levelname)s - %(filename)s: %(lineno)d] %(message)s'
    logging.basicConfig(format=log_msg_format,
                        level=logging.INFO,
                        handlers=[logging.FileHandler(filename=args.log_filename), logging.StreamHandler()])

    logger.info("Starting main process...")

    main(args)
