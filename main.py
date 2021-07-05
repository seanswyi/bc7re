import argparse
import os

from transformers import AutoConfig, AutoModel, AutoTokenizer

from model import DrugProtREModel
from trainer import Trainer


def main(args):
    args.train_file = os.path.join(args.data_dir, 'training', args.train_file)
    args.dev_file = os.path.join(args.data_dir, 'development', args.dev_file)

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    backbone_model = AutoModel.from_pretrained(args.model_name_or_path)

    model = DrugProtREModel(args=args, config=config, backbone_model=backbone_model, tokenizer=tokenizer)

    trainer = Trainer(args=args, config=config, model=model, tokenizer=tokenizer)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--bilinear_block_size', default=64, type=int)
    parser.add_argument('--data_dir', default='/hdd1/seokwon/data/BC7DP/drugprot-gs-training-development', type=str)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--dev_file', default='dev.json', type=str)
    parser.add_argument('--entity_marker', default='asterisk', type=str, choices=['asterisk', 'typed'])
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--max_seq_len', default=512, type=str)
    parser.add_argument('--model_name_or_path', default='dmis-lab/biobert-v1.1', type=str)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--num_labels', default=13, type=int)
    parser.add_argument('--train_file', default='train.json', type=str)

    args = parser.parse_args()

    main(args)
