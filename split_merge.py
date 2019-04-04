import pickle
from tqdm import tqdm
from parameters import *
import numpy as np
import scipy.spatial.distance
from evaluation import evaluation
from typing import *


with open(deprel2id_path, 'rb') as fin:
    deprel2id = pickle.load(fin)
with open(arg2id_path, 'rb') as fin:
    arg2id = pickle.load(fin)
with open(arghead2id_path, 'rb') as fin:
    arghead2id = pickle.load(fin)
with open(pos2id_path, 'rb') as fin:
    pos2id = pickle.load(fin)


init_key_dict = {(deprel, verbvoice, rela_position) :[]
    for deprel in deprel2id.keys()
    for verbvoice in ['a', 'p']
    for rela_position in ['l', 'r']}
init_arg_dict = {arg:[] for arg in arg2id.keys()}


class Cluster:
    BETA = 0
    GAMMA = 0
    def __init__(self, data: List[Tuple[str,str,str]]) -> None:
        self.data = data
        self.lex = np.array([0] * len(arghead2id))
        self.pos = np.array([0] * len(pos2id))
    def pos_sim(self, other):
        return 1 - scipy.spatial.distance.cosine(self.pos, other.pos)
    def lex_sim(self, other):
        return 1 - scipy.spatial.distance.cosine(self.lex, other.lex)
    def cons_sim(self, other):
        viol = 0
        i, j = 0, 0
        while i < len(self.data) and j < len(other.data):
            if self.data[i][0] == other.data[i][0] and self.data[i][1] == other.data[i][1]:
                viol += 1
                i += 1
                j += 1
            elif self.data[i] < other.data[j]:
                i += 1
            else:
                j += 1
        return 1 - 2 * viol / (len(self.data) + len(other.data))
    def score(self, other, beta, gamma):
        """ calculate the similarity between two clusters
        """
        if self.pos_sim(other) < Cluster.BETA:
            return 0
        elif self.cons_sim(other) < Cluster.GAMMA:
            return 0
        return self.lex_sim(other)
    def __iadd__(self, other):
        self.data.extend(other.data)
        self.data.sort()
        self.lex += other.lex
        self.pos += other.pos




def split_phase(flattened_data_path):
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
                        deprel = word_info[12]
                        arg = word_info[14]
                        idx = (word_info[0],word_info[1],word_info[4])
                        groundtruths[predicate][arg].append(idx)
                        predicts[predicate][(deprel,verbvoice,rela_position)].append(idx)
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
    assert(len(sentence)==0 and predicate is None)
    return groundtruths, predicts


def main():
    truths, predicts = split_phase(flattened_test_data_path)
    pre, coll, f1 = evaluation(truths, predicts)
    print(pre, coll, f1)

    # 1. combine all those verb's clusters together and calculate new result
    # truths_combine = {'combine': init_arg_dict.copy()}
    # predicts_combine = {'combine': init_key_dict.copy()}
    # for word in truths:
    #     for key in truths[word].keys():
    #         truths_combine['combine'][key] = truths_combine['combine'][key] | truths[word][key]
    # for word in predicts:
    #     for key in predicts[word].keys():
    #         predicts_combine['combine'][key] = predicts_combine['combine'][key] | predicts[word][key]
    # pre, coll, _, _ = eval_f1(truths_combine, predicts_combine)
    # print(pre, coll)

if __name__ == "__main__":
    main()
