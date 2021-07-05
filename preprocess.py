import argparse
import csv
import json
import os

from tqdm import tqdm


with open(file='/hdd1/seokwon/data/BC7DP/relation2id.json') as f:
    relation2id = json.load(fp=f)


def convert_data_to_features(data, tokenizer, entity_marker='asterisk', mode='train'):
    features = []

    pbar = tqdm(iterable=data, desc=f"Converting {mode} data to features", total=len(data))
    for sample in pbar:
        feature = {'input_ids': [],
                   'entity_positions': [],
                   'labels': []}

        title = sample['title']
        context = sample['context']
        text = ' '.join([title, context])

        entities = sample['entities']
        relations = sample['relations']

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

        # Insert entity markers in the front and back of entities.
        entity_starts = []
        entity_ends = []
        for entity_id, entity in entities.items():
            for mention in entity:
                entity_type = mention['type']
                entity_start = mention['start']
                entity_end = mention['end']
                entity_starts.append((entity_type, entity_start))
                entity_ends.append((entity_type, entity_end))

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
                    tokens_ = ['[' + entity_type + ']'] + tokens_

            if word_idx in [x[1] - 1 for x in entity_ends]:
                if entity_marker == 'asterisk':
                    tokens_ = tokens_ + ['*']
                elif entity_marker == 'typed':
                    idx = [x[1] for x in entity_ends].index(word_idx)
                    entity_type = entity_ends[idx][0]

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

        # Check whether the markers are correct.
        for entity_position in entity_positions:
            start = entity_position[0]
            end = entity_position[1]

            assert tokens[start:end][0] == tokens[start:end][-1] == '*', f"{tokens[start:end]}"

        # Create (head, tail, relation) triples.
        head_tail_pairs = []
        labels = []
        for relation in relations:
            label = [0] * len(relation2id)

            head_idx = int(relation['head_id'][1:]) - 1
            tail_idx = int(relation['tail_id'][1:]) - 1
            relation_id = relation['relation']

            relation_idx = relation2id[relation_id]
            label[relation_idx] = 1

            head_tail_pairs.append([head_idx, tail_idx])
            labels.append(label)

        # Build input IDs.
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        # Finalize features.
        feature['input_ids'] = input_ids
        feature['entity_positions'] = entity_positions
        feature['head_tail_pairs'] = head_tail_pairs
        feature['labels'] = labels

        features.append(feature)

    return features


def read_tsv(filename):
    with open(file=filename) as f:
        csv_reader = csv.reader(f, delimiter='\t', quotechar='"')
        data = [row for row in csv_reader]

    return data


def aggregate_data(abstracts, entities, relations=None, mode='train'):
    data = []

    pbar = tqdm(iterable=abstracts, desc=f"Aggregating data for {mode}", total=len(abstracts))
    for document in pbar:
        template = {'doc_id': str,
                    'title': str,
                    'context': str,
                    'entities': {},
                    'relations': []}

        doc_id = document[0]
        title = document[1]
        context = document[2]

        template['doc_id'] = doc_id
        template['title'] = title
        template['context'] = context

        doc_entities = [entity for entity in entities if entity[0] == doc_id]
        doc_relations = [relation for relation in relations if relation[0] == doc_id]

        for entity in doc_entities:
            entity_id = entity[1]
            entity_type = entity[2]
            entity_start = entity[3]
            entity_end = entity[4]
            entity_name = entity[5]

            entity_template = {'type': entity_type, 'name': entity_name, 'start': int(entity_start), 'end': int(entity_end)}

            try:
                template['entities'][entity_id].append(entity_template)
            except KeyError:
                template['entities'][entity_id] = [entity_template]

        for relation in doc_relations:
            relation_name = relation[1]
            head_entity = relation[2].split(':')[1]
            tail_entity = relation[3].split(':')[1]

            relation_template = {'head_id': head_entity, 'tail_id': tail_entity, 'relation': relation_name}
            template['relations'].append(relation_template)

        data.append(template)

    return data


def main(args):
    train_dir = os.path.join(args.data_dir, 'training')
    dev_dir = os.path.join(args.data_dir, 'development')

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

    train_data = aggregate_data(abstracts=train_abstracts, entities=train_entities, relations=train_relations)
    dev_data = aggregate_data(abstracts=dev_abstracts, entities=dev_entities, relations=dev_relations)

    train_save_file = os.path.join(train_dir, 'train.json')
    with open(file=train_save_file, mode='w') as f:
        json.dump(obj=train_data, fp=f, indent=2)

    dev_save_file = os.path.join(dev_dir, 'dev.json')
    with open(file=dev_save_file, mode='w') as f:
        json.dump(obj=dev_data, fp=f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='/hdd1/seokwon/data/BC7DP/drugprot-gs-training-development', type=str)
    parser.add_argument('--abstracts_file', default='drugprot_MODE_abstracs.tsv', type=str)
    parser.add_argument('--entities_file', default='drugprot_MODE_entities.tsv', type=str)
    parser.add_argument('--relations_file', default='drugprot_MODE_relations.tsv', type=str)

    args = parser.parse_args()

    main(args)
