import pickle
from tqdm import tqdm
from parameters import *

with open(deprel2id_path, 'rb') as fin:
    deprel2id = pickle.load(fin)
with open(arg2id_path, 'rb') as fin:
    arg2id = pickle.load(fin)
with open(preposition2id_path, 'rb') as fin:
    preposition2id = pickle.load(fin)

prepositions = []

init_key_dict = {(deprel, verbvoice, rela_position, preposition) :set() 
    for deprel in deprel2id.keys()
    for verbvoice in ['a', 'p'] 
    for rela_position in ['l', 'r']
    for preposition in preposition2id.keys()}
init_arg_dict = {arg:set() for arg in arg2id.keys()}



def get_label(flattened_data_path):
    groundtruths = dict()
    predicts = dict()
    sentences = []
    with open(flattened_data_path, 'r') as fin:
        sentences = fin.readlines()


    sentence = []
    predicate = None
    predicate_id = -1
    for line in tqdm(sentences):
        word_info = line.strip().split()
        if len(word_info) == 0:
            if predicate is not None:
                for word_info in sentence:
                    if word_info[-1] != '_':
                        rela_position = 'r' if int(word_info[4]) > predicate_id else 'l'
                        verbvoice = 'p' if sentence[predicate_id-1][8] == 'VBN' else 'a'
                        preposition = '<PAD>' if word_info[8] != 'IN' else word_info[6]
                        deprel = word_info[12]
                        arg = word_info[14]
                        idx = (word_info[0],word_info[1],word_info[4])
                        groundtruths[predicate][arg].add(idx)
                        predicts[predicate][(deprel,verbvoice,rela_position,preposition)].add(idx)
            sentence = []
            predicate = None
            predicate_id = -1
        else:
            if word_info[3] == '1' and 'V' == word_info[8][0]:
                predicate = word_info[6]
                predicate_id = int(word_info[4])
                if predicate not in groundtruths:
                    groundtruths[predicate] = init_arg_dict.copy()
                    predicts[predicate] = init_key_dict.copy()
            sentence.append(word_info)
    if len(sentence) != 0 and predicate is not None:
        for word_info in sentence:
            if word_info[-1] != '_':
                        rela_position = 'r' if int(word_info[4]) > predicate_id else 'l'
                        verbvoice = 'p' if sentence[predicate_id-1][8] == 'VBN' else 'a'
                        preposition = '<PAD>' if word_info[8] != 'IN' else word_info[6]
                        deprel = word_info[12]
                        arg = word_info[14]
                        idx = (word_info[0],word_info[1],word_info[4])
                        groundtruths[predicate][arg].add(idx)
                        predicts[predicate][(deprel,verbvoice,rela_position,preposition)].add(idx)
    return groundtruths, predicts

def main():
    truths, predicts = get_label(flattened_test_data_path)
    from evaluation import eval_f1
    pre, coll, _, _ = eval_f1(truths, predicts)
    print(pre, coll)

    truths_combine = {'combine': init_arg_dict.copy()}
    predicts_combine = {'combine': init_key_dict.copy()}
    for word in truths:
        for key in truths[word].keys():
            truths_combine['combine'][key] = truths_combine['combine'][key] | truths[word][key]
    for word in predicts:
        for key in predicts[word].keys():
            predicts_combine['combine'][key] = predicts_combine['combine'][key] | predicts[word][key]
    
    pre, coll, _, _ = eval_f1(truths_combine, predicts_combine)
    print(pre, coll)

if __name__ == "__main__":
    main()
