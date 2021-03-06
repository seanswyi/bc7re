import argparse
import csv
from itertools import product
import json
import math
import os
import sys

import stanza
from tqdm import tqdm


MAX_INT = sys.maxsize
while True:
    try:
        csv.field_size_limit(MAX_INT)
        break
    except OverflowError:
        MAX_INT = int(MAX_INT / 10)


with open(file='/hdd1/seokwon/data/BC7DP/relation2id.json') as f:
    relation2id = json.load(fp=f)


def convert_data_to_features(data, tokenizer, negative_ratio=2, entity_marker='asterisk', mode='train', setting='document'):
    features = []

    num_positive_samples = 0
    num_negative_samples = 0

    pbar = tqdm(iterable=data, desc=f"Converting {setting} {mode} data to features", total=len(data))
    for sample in pbar:
        feature = {'input_ids': [],
                   'entity_positions': [],
                   'labels': []}

        doc_id = sample['pmid']
        title = sample['title']
        context = sample['context']
        text = ' '.join([title, context])

        entities = sample['entities']
        relations = sample['relations']
        pair2relation = {(x['head_id'], x['tail_id']): x['relation'] for x in relations}

        # Sort entities by their starting positions to make our lives a little easier.
        entities_by_position = [(entity_id, entity) for entity_id, entity in entities.items()]
        entities_by_position = sorted(entities_by_position, key=lambda x: x[1][0]['start'])

        # Add in new whitespace after punctuation to make splitting easier.
        new_text = []
        current_idx = 0
        for word in text.split():
            if (len(word) > 1) and (word[-1] in ['.', ',', ';', ':']):
                new_text.append(word[:-1] + ' ' + word[-1])
                num_added_spaces = 1
                current_idx = len(' '.join(new_text))
            else:
                num_added_spaces = 0
                new_text.append(word)

            for entity_idx, entity in enumerate(entities_by_position):
                for mention_idx, mention in enumerate(entity[1]):
                    start = mention['start']
                    end = mention['end']

                    if start >= current_idx:
                        start += num_added_spaces
                        end += num_added_spaces

                    entities_by_position[entity_idx][1][mention_idx]['start'] = start
                    entities_by_position[entity_idx][1][mention_idx]['end'] = end

        new_text = ' '.join(new_text).split()

        # Create a mapping between word-level and character-level spans.
        word2char_span = {}
        start_idx = 0
        for idx, word in enumerate(new_text):
            char_start = start_idx
            char_end = char_start + len(word)
            word2char_span[idx] = (char_start, char_end)
            start_idx += len(word) + 1

        for entity_id, entity in entities_by_position:
            adjusted_mentions = []
            for mention in entity:
                mention_start = mention['start']
                mention_end = mention['end']

                for word_span, char_span in word2char_span.items():
                    span_range = list(range(char_span[0], char_span[1] + 1))

                    if mention_start in span_range:
                        word_lvl_start = word_span

                    if mention_end in span_range:
                        word_lvl_end = word_span + 1

                mention['start'] = word_lvl_start
                mention['end'] = word_lvl_end

                adjusted_mentions.append(mention)

            entities[entity_id] = adjusted_mentions

        entity_starts = []
        entity_ends = []
        for entity_id, entity in entities.items():
            for mention in entity:
                entity_type = mention['type']
                entity_start = mention['start']
                entity_end = mention['end']
                entity_starts.append((entity_type, entity_start))
                entity_ends.append((entity_type, entity_end))

        # Create entity pairs.
        chemical_entities = []
        gene_entities = []
        for entity_id, entity in entities.items():
            for mention in entity:
                if mention['type'] == 'CHEMICAL':
                    chemical_entities.append(entity_id)
                    break
                elif mention['type'] in ['GENE', 'GENE-Y', 'GENE-N']:
                    gene_entities.append(entity_id)
                    break

        if (chemical_entities == []) or (gene_entities == []):
            continue

        all_entity_pairs = list(product(chemical_entities, gene_entities))

        # if mode == 'train':
        temp_pairs = []
        for pair in all_entity_pairs:
            head_id = pair[0]
            tail_id = pair[1]

            head_sentence_id = entities[head_id][0]['sentence_id']
            tail_sentence_id = entities[tail_id][0]['sentence_id']

            if head_sentence_id == tail_sentence_id:
                temp_pairs.append(pair)

        all_entity_pairs = temp_pairs

        # Create (head, tail, relation) triples.
        head_tail_pairs = []
        sentence_ids = []
        labels = []
        total_num_negative = math.floor(len(relations) * negative_ratio)
        negative_count = 0
        for pair in all_entity_pairs:
            relation = [0] * len(relation2id)

            head_id = pair[0]
            head_idx = int(head_id[1:]) - 1
            head_sentence_id = entities[head_id][0]['sentence_id']

            tail_id = pair[1]
            tail_idx = int(tail_id[1:]) - 1
            tail_sentence_id = entities[tail_id][0]['sentence_id']

            if (head_id, tail_id) in pair2relation:
                relation_name = pair2relation[(head_id, tail_id)]
                relation_id = relation2id[relation_name]
                relation[relation_id] = 1

                num_positive_samples += 1
            else:
                if (mode == 'train') and (negative_count == total_num_negative):
                    continue

                relation[0] = 1
                negative_count += 1

                num_negative_samples += 1

            head_tail_pairs.append([head_idx, tail_idx])
            sentence_ids.append([head_sentence_id, tail_sentence_id])

            labels.append(relation)

        if sum(head_tail_pairs, []) == []:
            continue

        if setting == 'document':
            tokens = []
            word2token_span = {}
            current_idx = 0
            for word_idx, word in enumerate(new_text):
                tokens_ = tokenizer.tokenize(word)
                token_start = len(tokens)

                if word_idx in [x[1] for x in entity_starts]:
                    if entity_marker == 'asterisk':
                        tokens_ = ['*'] + tokens_
                    elif entity_marker == 'typed':
                        idx = [x[1] for x in entity_starts].index(word_idx)
                        entity_type = entity_starts[idx][0]

                        if 'GENE' in entity_type:
                            entity_type = 'GENE'

                        tokens_ = ['[' + entity_type + ']'] + tokens_

                if word_idx in [x[1] - 1 for x in entity_ends]:
                    if entity_marker == 'asterisk':
                        tokens_ = tokens_ + ['*']
                    elif entity_marker == 'typed':
                        idx = [x[1] - 1 for x in entity_ends].index(word_idx)
                        entity_type = entity_ends[idx][0]

                        if 'GENE' in entity_type:
                            entity_type = 'GENE'

                        tokens_ = tokens_ + ['[' + entity_type + ']']

                tokens.extend(tokens_)
                token_end = len(tokens)
                token_span = list(range(token_start, token_end + 1))

                word2token_span[word_idx] = token_span

            # Convert word-level spans to token-level spans.
            entity_positions = []
            for entity_id, entity in entities.items():
                for mention in entity:
                    word_lvl_start = mention['start']
                    word_lvl_end = mention['end']

                    if word_lvl_end - word_lvl_start == 1:
                        token_lvl_start = word2token_span[word_lvl_start][0]
                        token_lvl_end = word2token_span[word_lvl_start][-1]
                    elif word_lvl_end - word_lvl_start > 1:
                        token_lvl_start = word2token_span[word_lvl_start][0]
                        token_lvl_end = word2token_span[word_lvl_end - 1][-1]

                    entity_positions.append([token_lvl_start, token_lvl_end])

            # Build input IDs.
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

            feature = {'pmid': doc_id,
                       'input_ids': input_ids,
                       'entity_positions': entity_positions,
                       'head_tail_pairs': head_tail_pairs,
                       'sentence_ids': sentence_ids,
                       'labels': labels}
            features.append(feature)
        elif setting == 'sentence':
            import pdb; pdb.set_trace()
            for pair, label in zip(head_tail_pairs, labels):
                tokens = []
                word2token_span = {}
                current_idx = 0

                new_entity_starts = [position for idx, position in enumerate(entity_starts) if idx in pair]
                new_entity_ends = [position for idx, position in enumerate(entity_ends) if idx in pair]

                for word_idx, word in enumerate(new_text):
                    tokens_ = tokenizer.tokenize(word)
                    token_start = len(tokens)

                    if word_idx in [x[1] for x in new_entity_starts]:
                        if entity_marker == 'asterisk':
                            tokens_ = ['*'] + tokens_
                        elif entity_marker == 'typed':
                            idx = [x[1] for x in new_entity_starts].index(word_idx)
                            entity_type = new_entity_starts[idx][0]

                            if 'GENE' in entity_type:
                                entity_type = 'GENE'

                            tokens_ = ['[' + entity_type + ']'] + tokens_

                    if word_idx in [x[1] - 1 for x in new_entity_ends]:
                        if entity_marker == 'asterisk':
                            tokens_ = tokens_ + ['*']
                        elif entity_marker == 'typed':
                            idx = [x[1] - 1 for x in new_entity_ends].index(word_idx)
                            entity_type = new_entity_ends[idx][0]

                            if 'GENE' in entity_type:
                                entity_type = 'GENE'

                            tokens_ = tokens_ + ['[' + entity_type + ']']

                    tokens.extend(tokens_)
                    token_end = len(tokens)
                    token_span = list(range(token_start, token_end + 1))

                    word2token_span[word_idx] = token_span

                # Convert word-level spans to token-level spans.
                entity_positions = []
                for entity_id, entity in entities.items():
                    for mention in entity:
                        word_lvl_start = mention['start']
                        word_lvl_end = mention['end']

                        if word_lvl_end - word_lvl_start == 1:
                            token_lvl_start = word2token_span[word_lvl_start][0]
                            token_lvl_end = word2token_span[word_lvl_start][-1]
                        elif word_lvl_end - word_lvl_start > 1:
                            token_lvl_start = word2token_span[word_lvl_start][0]
                            token_lvl_end = word2token_span[word_lvl_end - 1][-1]

                        entity_positions.append([token_lvl_start, token_lvl_end])

                # Build input IDs.
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

                feature = {'pmid': doc_id,
                           'input_ids': input_ids,
                           'entity_positions': entity_positions,
                           'head_tail_pairs': [pair],
                           'sentence_ids': sentence_ids,
                           'labels': label}
                features.append(feature)

    print(f"Number of positive samples ({mode}): {num_positive_samples}")
    print(f"Number of negative samples ({mode}): {num_negative_samples}")

    return features


def read_tsv(filename):
    with open(file=filename) as f:
        csv_reader = csv.reader(f, delimiter='\t', quotechar='"')
        data = [row for row in csv_reader]

    return data


def aggregate_data(stanza_pipeline, abstracts, entities, relations=[], mode='train', data_type='bc7dp'):
    data = []

    pbar = tqdm(iterable=abstracts, desc=f"Aggregating data for {mode}", total=len(abstracts))
    for document in pbar:
        template = {'pmid': str,
                    'title': str,
                    'context': str,
                    'entities': {},
                    'relations': []}

        doc_id = document[0]
        title = document[1]
        context = document[2]

        full_text = ' '.join([title, context])
        stanza_output = stanza_pipeline(full_text)

        sentenceid2charspan = {}
        for idx, sentence in enumerate(stanza_output.sentences):
            start_char = sentence.words[0].misc.split('|')[0].split('=')[-1]
            end_char = sentence.words[-1].misc.split('|')[-1].split('=')[-1]
            sentenceid2charspan[idx] = list(range(int(start_char), int(end_char) + 1))

        template['pmid'] = doc_id
        template['title'] = title
        template['context'] = context

        doc_entities = [entity for entity in entities if entity[0] == doc_id]
        doc_relations = [relation for relation in relations if relation[0] == doc_id]

        for entity in doc_entities:
            entity_id = entity[1]
            entity_type = entity[2]
            entity_start = int(entity[3])
            entity_end = int(entity[4])
            entity_name = entity[5]

            for sentence_id, positions in sentenceid2charspan.items():
                if entity_start in positions:
                    entity_sentence_id = sentence_id
                    break

            entity_template = {'type': entity_type, 'name': entity_name, 'start': int(entity_start), 'end': int(entity_end), 'sentence_id': entity_sentence_id}

            try:
                template['entities'][entity_id].append(entity_template)
            except KeyError:
                template['entities'][entity_id] = [entity_template]

        for relation in doc_relations:
            if (data_type == 'bc7dp') and (mode != 'test'):
                relation_name = relation[1]
                head_entity = relation[2].split(':')[1]
                tail_entity = relation[3].split(':')[1]
            elif data_type == 'bc6cp':
                relation_name = relation[3]
                head_entity = relation[4].split(':')[1]
                tail_entity = relation[5].split(':')[1]

            relation_template = {'head_id': head_entity, 'tail_id': tail_entity, 'relation': relation_name}
            template['relations'].append(relation_template)

        data.append(template)

    return data


def main(args):
    print("Arguments:")
    for name, value in vars(args).items():
        print(f'\t{name}: {value}')

    stanza.download('en', package='craft')
    stanza_pipeline = stanza.Pipeline('en', package='craft')

    if args.data_type == 'bc7dp':
        train_dir = os.path.join(args.data_dir, 'training')
        dev_dir = os.path.join(args.data_dir, 'development')
        test_dir = os.path.join(args.data_dir, 'test-background')
    elif args.data_type == 'bc6cp':
        train_dir = os.path.join(args.data_dir, 'chemprot_training')
        dev_dir = os.path.join(args.data_dir, 'chemprot_development')
        test_dir = os.path.join(args.data_dir, 'chemprot_test_gs')

    train_abstracts_file = os.path.join(train_dir, args.abstracts_file.replace('MODE', 'training'))
    train_entities_file = os.path.join(train_dir, args.entities_file.replace('MODE', 'training'))
    train_relations_file = os.path.join(train_dir, args.relations_file.replace('MODE', 'training'))

    dev_abstracts_file = os.path.join(dev_dir, args.abstracts_file.replace('MODE', 'development'))
    dev_entities_file = os.path.join(dev_dir, args.entities_file.replace('MODE', 'development'))
    dev_relations_file = os.path.join(dev_dir, args.relations_file.replace('MODE', 'development'))

    train_abstracts = read_tsv(filename=train_abstracts_file)
    train_entities = read_tsv(filename=train_entities_file)
    train_relations = read_tsv(filename=train_relations_file)

    dev_abstracts = read_tsv(filename=dev_abstracts_file)
    dev_entities = read_tsv(filename=dev_entities_file)
    dev_relations = read_tsv(filename=dev_relations_file)

    train_data = aggregate_data(stanza_pipeline=stanza_pipeline, abstracts=train_abstracts, entities=train_entities, relations=train_relations, data_type=args.data_type)
    dev_data = aggregate_data(stanza_pipeline=stanza_pipeline, abstracts=dev_abstracts, entities=dev_entities, relations=dev_relations, mode='dev', data_type=args.data_type)

    train_save_file = os.path.join(train_dir, f'train_{args.data_type}.json')
    with open(file=train_save_file, mode='w') as f:
        json.dump(obj=train_data, fp=f, indent=2)

    dev_save_file = os.path.join(dev_dir, f'dev_{args.data_type}.json')
    with open(file=dev_save_file, mode='w') as f:
        json.dump(obj=dev_data, fp=f, indent=2)

    if args.data_type == 'bc6cp':
        test_abstracts_file = os.path.join(test_dir, args.abstracts_file.replace('MODE', 'test')).replace('.tsv', '_gs.tsv')
        test_entities_file = os.path.join(test_dir, args.entities_file.replace('MODE', 'test')).replace('.tsv', '_gs.tsv')
        test_relations_file = os.path.join(test_dir, args.relations_file.replace('MODE', 'test')).replace('.tsv', '_gs.tsv')

        test_abstracts = read_tsv(filename=test_abstracts_file)
        test_entities = read_tsv(filename=test_entities_file)
        test_relations = read_tsv(filename=test_relations_file)

        test_data = aggregate_data(stanza_pipeline=stanza_pipeline, abstracts=test_abstracts, entities=test_entities, relations=test_relations, mode='test', data_type=args.data_type)

        test_save_file = os.path.join(test_dir, f'test_{args.data_type}.json')
        with open(file=test_save_file, mode='w') as f:
            json.dump(obj=test_data, fp=f, indent=2)
    elif args.data_type == 'bc7dp':
        test_abstracts_file = os.path.join(test_dir, args.abstracts_file.replace('drugprot_MODE', 'test_background')).replace('abstracs', 'abstracts')
        test_entities_file = os.path.join(test_dir, args.entities_file.replace('drugprot_MODE', 'test_background'))

        test_abstracts = read_tsv(filename=test_abstracts_file)
        test_entities = read_tsv(filename=test_entities_file)

        test_data = aggregate_data(stanza_pipeline=stanza_pipeline, abstracts=test_abstracts, entities=test_entities, mode='test', data_type=args.data_type)

        test_save_file = os.path.join(test_dir, f'test_{args.data_type}.json')
        with open(file=test_save_file, mode='w') as f:
            json.dump(obj=test_data, fp=f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='/hdd1/seokwon/data/BC7DP/drugprot-gs-training-development', type=str)
    parser.add_argument('--data_type', default='bc7dp', type=str, choices=['bc6cp', 'bc7dp'])
    parser.add_argument('--abstracts_file', default='drugprot_MODE_abstracs.tsv', type=str)
    parser.add_argument('--entities_file', default='drugprot_MODE_entities.tsv', type=str)
    parser.add_argument('--relations_file', default='drugprot_MODE_relations.tsv', type=str)

    args = parser.parse_args()

    if args.data_type == 'bc6cp':
        args.data_dir = '/hdd1/seokwon/data/BC6CP/ChemProt_Corpus'
        args.abstracts_file = 'chemprot_MODE_abstracts.tsv'
        args.entities_file = 'chemprot_MODE_entities.tsv'
        args.relations_file = 'chemprot_MODE_relations.tsv'

    main(args)
