import argparse
import csv
import json
import os

from tqdm import tqdm


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

            entity_template = {'type': entity_type, 'name': entity_name, 'start': entity_start, 'end': entity_end}

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
