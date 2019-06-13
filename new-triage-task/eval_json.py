#!/usr/bin/env python

### Collect gold standard and predicted annotations from BioC &
### Compute classification performance scores

import json
import sys
import operator

def JSON_Document(document):
    confidence = 0
    id = document['id']

    if 'infons' not in document:
        label = 'no'
    else:
        infons = document['infons']
        if 'relevant' in infons:
            label = infons['relevant']
            label = label.lower()
            if 'confidence' in infons:
                confidence = float(infons['confidence'])
        else:
            label = 'no'

    if label != 'yes':
        if confidence == 0:
            confidence = -1
        else:
            confidence = -confidence

    if 'relations' not in document:
        relations = []
    else:
        relations = document['relations']

    return id, label, confidence, relations

def JSON_Collection_Triage(collection):
    positives = set()
    negatives = set()

    for document in collection.get('documents', []):
        id, label, confidence, relations = JSON_Document(document)
        if label == 'yes':
            positives.add(id)
        else:
            negatives.add(id)

    return positives, negatives

def Classification_Performance_Triage(collection, gold_standard_positive, gold_standard_negative):
    correct = prediction_count = 0
    precision = recall = f1 = 0
    tp = fp = tn = fn = 0

    previously_seen = set()
    prediction_dict = {}

    for document in collection.get('documents', []):
        id, label, confidence, relations = JSON_Document(document)
        if (id not in previously_seen) and (id in gold_standard_positive or id in gold_standard_negative):
            if label == 'yes':
                if id in gold_standard_positive:
                    correct += 1.
                prediction_count += 1.
            prediction_dict[id] = confidence
            previously_seen.add(id)

    if prediction_count > 0 and correct > 0 and len(gold_standard_positive) > 0:
        precision = correct / prediction_count
        recall = correct / len(gold_standard_positive)
        f1 = 2. * precision * recall / (precision + recall)

        tp = correct
        fp = prediction_count - correct
        tn = len(gold_standard_negative) - fp
        fn = len(gold_standard_positive) - correct

    correct = prediction_count = 0
    average_precision = 0

    prediction_dict = sorted(prediction_dict.items(), key=operator.itemgetter(1), reverse=True)
    for id, confidence in prediction_dict:
        prediction_count += 1.
        if id in gold_standard_positive:
            correct += 1.
            average_precision += correct / prediction_count
    average_precision /= len(gold_standard_positive)

    return average_precision, precision, recall, f1, tp, fp, tn, fn

def JSON_Collection_Relation(collection):
    all_ids = set()
    all_relations = set()
 
    for document in collection.get('documents', []):
        id, label, confidence, relations = JSON_Document(document)
        for relation in relations:
            if 'infons' in relation:
                relation_flag = 0
                infon_values = []

                infons = relation['infons']
                for infon_type in infons:
                    infon_type_lowercase = infon_type.lower()
                    if infon_type_lowercase == 'relation':
                        relation_flag = 1
                    elif infon_type_lowercase[:4] == 'gene':
                        infon_values.append(infons[infon_type])

                if relation_flag == 1:
                    infon_values.sort()
                    relation_string = 'PMID' + id + '_' + '_'.join(infon_values)
                    all_ids.add(id)
                    all_relations.add(relation_string)

    return all_ids, all_relations

def PMID_Relation_Count(substring, relations):
    count = 0

    for relation in relations:
        if relation.startswith(substring):
            count += 1.

    return count

def Classification_Performance_Relation(collection, gold_standard_ids, gold_standard_relations):
    correct = prediction_count = 0
    micro_precision = micro_recall = micro_f1 = 0
    macro_precision = macro_recall = macro_f1 = 0

    previously_seen = set()
    prediction_dict = {}

    for document in collection.get('documents', []):
        id, label, confidence, relations = JSON_Document(document)
        if id in gold_standard_ids:
            each_correct = each_prediction_count = 0
            precision = recall = f1 = 0

            for relation in relations:
                if 'infons' in relation:
                    relation_flag = 0
                    infon_values = []
                    relation_confidence = 0

                    infons = relation['infons']
                    for infon_type in infons:
                        infon_type_lowercase = infon_type.lower()
                        if infon_type_lowercase == 'relation':
                            relation_flag = 1
                        elif infon_type_lowercase[:4] == 'gene':
                            infon_values.append(infons[infon_type])
                        elif infon_type_lowercase == 'confidence':
                            relation_confidence = float(infons[infon_type])
            
                    if relation_flag == 1:
                        infon_values.sort()
                        relation_string = 'PMID' + id + '_' + '_'.join(infon_values)
                        if relation_string not in previously_seen:
                            if relation_string in gold_standard_relations:
                                correct += 1.
                                each_correct += 1.
                            prediction_count += 1.
                            each_prediction_count += 1.
                            prediction_dict[relation_string] = relation_confidence
                            previously_seen.add(relation_string)

            relation_count = PMID_Relation_Count('PMID' + id + '_', gold_standard_relations)
            if each_prediction_count > 0 and each_correct > 0 and relation_count > 0:
                precision = each_correct / each_prediction_count
                recall = each_correct / relation_count
                f1 = 2. * precision * recall / (precision + recall)
            
            macro_precision += precision
            macro_recall += recall
            macro_f1 += f1

    if prediction_count > 0 and correct > 0 and len(gold_standard_relations) > 0:
        micro_precision = correct / prediction_count
        micro_recall = correct / len(gold_standard_relations)
        micro_f1 = 2. * micro_precision * micro_recall / (micro_precision + micro_recall)

        macro_precision /= len(gold_standard_ids)
        macro_recall /= len(gold_standard_ids)
        macro_f1 /= len(gold_standard_ids)
 
    return micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1

program_name = subtask = gold_standard_file = prediction_file = None
if len(sys.argv) == 4:
    program_name, subtask, gold_standard_file, prediction_file = sys.argv
else:
    sys.exit('Usage: ' + sys.argv[0] + ' [triage|relation] [gold_standard_file] [prediction_file]')

if subtask != 'triage' and subtask != 'relation':
    sys.exit('Usage: ' + sys.argv[0] + ' [triage|relation] [gold_standard_file] [prediction_file]')

gold_standard_collection = prediction_collection = None

with open(gold_standard_file) as f1, open(prediction_file) as f2:
    gold_standard_collection = json.load(f1)
    prediction_collection = json.load(f2)

    if subtask == 'triage':
        gold_standard_positive, gold_standard_negative = JSON_Collection_Triage(gold_standard_collection)
        average_precision, precision, recall, f1, tp, fp, tn, fn = Classification_Performance_Triage(prediction_collection, gold_standard_positive, gold_standard_negative)

        print('Avg Precision: %.4f' % average_precision)
        print('TP: %d / FP: %d / TN: %d / FN: %d' % (tp, fp, tn, fn))
        print('Precision: %.4f' % precision)
        print('Recall: %.4f' % recall)
        print('F1: %.4f' % f1)        
    else:
        gold_standard_ids, gold_standard_relations = JSON_Collection_Relation(gold_standard_collection)
        micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1 = Classification_Performance_Relation(prediction_collection, gold_standard_ids, gold_standard_relations)

        print('Micro Precision: %.4f' % micro_precision)
        print('Micro Recall: %.4f' % micro_recall)
        print('Micro F1: %.4f' % micro_f1)
        print('Macro Precision: %.4f' % macro_precision)
        print('Macro Recall: %.4f' % macro_recall)
        print('Macro F1: %.4f' % macro_f1)
