import pickle
from tqdm import tqdm
import time
from copy import deepcopy
from evaluation import evaluation
from parameters import *


with open(deprel2id_path, 'rb') as fin:
    deprel2id = pickle.load(fin)
with open(arg2id_path, 'rb') as fin:
    arg2id = pickle.load(fin)

init_deprel_dict = {deprel:[] for deprel in deprel2id.keys()}
init_arg_dict = {arg:[] for arg in arg2id.keys()}

def get_label(flattened_data_path):
    groundtruths = dict()
    predicts = dict()
    sentences = []
    with open(flattened_data_path, 'r') as fin:
        sentences = fin.readlines()
    # cluster arguments according to their dependency relation to their governor
    sentence = []
    predicate = None
    for line in tqdm(sentences):
        word_info = line.strip().split()
        if len(word_info) == 0:
            if predicate is not None:
                for word_info in sentence:
                    if (is_normal_argument(word_info[FLATTEN2ID[SR]])):
                        deprel = word_info[FLATTEN2ID[DEPREL]]
                        arg = word_info[FLATTEN2ID[SR]]
                        idx = (word_info[FLATTEN2ID[SENT_ID]],word_info[FLATTEN2ID[PREDICATE_ID]],word_info[FLATTEN2ID[WORD_ID]])
                        groundtruths[predicate][arg].append(idx)
                        predicts[predicate][deprel].append(idx)
            sentence = []
            predicate = None
        else:
            if is_predicate(word_info[FLATTEN2ID[IS_PREDICATE]]) and is_verb(word_info[FLATTEN2ID[POS]]):
                predicate = word_info[6]
                if predicate not in groundtruths:
                    groundtruths[predicate] = deepcopy(init_arg_dict)
                    predicts[predicate] = deepcopy(init_deprel_dict)
            sentence.append(word_info)
    assert(len(sentence) == 0 and predicate is None)
    return groundtruths, predicts

def main():
    truths, predicts = get_label(flattened_test_data_path)
    pre, coll, f1 = evaluation(truths, predicts)
    print(pre, coll, f1)
    # set_truths, set_predicts = dict(), dict()
    # for word in tqdm(truths.keys()):
    #     set_truths[word] = dict()
    #     set_predicts[word] = dict()
    #     for truth_label in truths[word].keys():
    #         set_truths[word][truth_label] = set(truths[word][truth_label])
    #     for pre_label in predicts[word].keys():
    #         set_predicts[word][pre_label] = set(predicts[word][pre_label])
    # pre, coll, _, _ = eval_f1(set_truths, set_predicts)

    # truths_combine = {'combine': init_arg_dict.copy()}
    # predicts_combine = {'combine': init_deprel_dict.copy()}
    # for word in truths:
    #     for key in truths[word].keys():
    #         truths_combine['combine'][key] = truths_combine['combine'][key] + truths[word][key]
    # for word in predicts:
    #     for key in predicts[word].keys():
    #         predicts_combine['combine'][key] = predicts_combine['combine'][key] + predicts[word][key]
    
    # pre, coll, f1 = evaluation(truths_combine, predicts_combine)
    # print(pre, coll, f1)

if __name__ == "__main__":
    main()
