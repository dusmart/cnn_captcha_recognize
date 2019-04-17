import pickle
from tqdm import tqdm
from parameters import *
import numpy as np
import scipy.spatial.distance
from evaluation import evaluation
from typing import *
from data_utils import _UNK_, _PAD_, _ROOT_, _NUM_, is_number
from copy import deepcopy

def find_argument_head(sentence: List[List[str]], word_info: List[str]) -> List[str]:
    """ find a child for given word in sentence, return None if not exists
    """
    if not is_preposition(word_info[8]):
        return word_info
    for i in range(len(sentence)-1, -1, -1):
        if sentence[i][10] == word_info[4]:
            return find_argument_head(sentence, sentence[i])
    return word_info


with open(deprel2id_path, 'rb') as fin:
    deprel2id = pickle.load(fin)
with open(arg2id_path, 'rb') as fin:
    arg2id = pickle.load(fin)
with open(arghead2id_path, 'rb') as fin:
    arghead2id = pickle.load(fin)
with open(pos2id_path, 'rb') as fin:
    pos2id = pickle.load(fin)
with open(pretrained2id_path, 'rb') as fin:
    pretrained2id = pickle.load(fin)
with open(pretrained_embed_path, 'rb') as fin:
    embedding = pickle.load(fin)
with open(preposition2id_path, 'rb') as fin:
    preposition2id = pickle.load(fin)

class Cluster:
    BETA = 0
    GAMMA = 0
    LEX_GROUP = 5
    DELTA = 1
    def __init__(self, data: List[Tuple[str,str,str]]) -> None:
        self.data = data
        self.lex = np.zeros((self.LEX_GROUP, pretrained_emb_size), dtype=np.float32)
        self.lex_weight = np.zeros((self.LEX_GROUP, ), np.float32)
        self.pos = np.array([0.0] * len(pos2id))
        self.left_right = np.array([0] * 2)
        self.verb_voice = np.array([0] * 2)
        self.relation = np.array([0] * len(deprel2id))
    def pos_sim(self, other) -> float:
        return 1 - scipy.spatial.distance.cosine(self.pos, other.pos)
    def lex_sim(self, other) -> float:
        scores = np.dot(self.lex, np.transpose(other.lex))
        for i in range(self.LEX_GROUP):
            norm1 = np.linalg.norm(self.lex[i, :])
            norm2 = np.linalg.norm(other.lex[i, :])
            if abs(norm1) > 0.00001:
                scores[i, :] /= norm1
            if abs(norm2) >= 0.00001:
                scores[:, i] /= norm2
        weights = np.dot(np.transpose(self.lex_weight), other.lex_weight)
        #weights /= np.sum(np.sum(weights))
        weighted_scores = np.multiply(scores, weights)
        return scores[np.unravel_index(np.argmax(weighted_scores, axis=None), weighted_scores.shape)]

    def syn_sim(self, other) -> float:
        position = 1 - scipy.spatial.distance.cosine(self.left_right, other.left_right)
        voice = 1 - scipy.spatial.distance.cosine(self.verb_voice, other.verb_voice)
        relation = 1 - scipy.spatial.distance.cosine(self.relation, other.relation)
        return (position + voice + relation) / 3
    def cons_sim(self, other) -> float:
        viol = 0
        i, j, i_, j_ = 0, 0, 0, 0
        while i < len(self.data) and j < len(other.data):
            viol += int(self.data[i][0] == other.data[j][0] and self.data[i][1] == other.data[j][1])
            i_ = int(self.data[i][0] <= other.data[j][0] or self.data[i][1] <= other.data[j][1])
            j_ = int(self.data[i][0] >= other.data[j][0] and self.data[i][1] >= other.data[j][1])
            i += i_
            j += j_
        return 1 - 2 * viol / (len(self.data) + len(other.data))
    def score(self, other) -> float:
        """ calculate the similarity between two clusters
        """
        if len(self) == 0 or len(other) == 0:
            return 1
        if self.pos_sim(other) < Cluster.BETA:
            return 0
        elif self.cons_sim(other) < Cluster.GAMMA:
            return 0
        return self.lex_sim(other) * self.DELTA + self.pos_sim(other) * (1-self.DELTA)
    def __iadd__(self, other):
        assert(len(self) >= len(other))
        for i in range(self.LEX_GROUP):
            if other.lex_weight[i] > 0.999:
                self.merge_lex(other.lex[i], other.lex_weight[i])
        self.pos += other.pos
        self.data += other.data
        self.left_right += other.left_right
        self.relation += other.relation
        self.verb_voice += other.verb_voice
        self.data.sort()
        return self
    def __len__(self):
        return len(self.data)
    def merge_lex(self, embed, weight, target=None):
        if target is not None:
            self.lex[target, :] = (self.lex[target, :] * self.lex_weight[target] + embed * weight) / (self.lex_weight[target]+weight)
            self.lex_weight[target] += weight
        else:
            candidate_score = np.dot(self.lex[0, :], np.array(embed))
            target = 0
            for i in range(1, self.LEX_GROUP):
                score = np.dot(self.lex[i], embed)
                if abs(self.lex_weight[i]) < 0.001:
                    target = i
                    break
                elif score > candidate_score:
                    target = i
                    score = candidate_score
            self.merge_lex(embed, weight, target)
    def append(self, value, pos, arghead, left_right, verb_voice, deprel):
        if arghead in pretrained2id:
            idx = pretrained2id[arghead]
            embed = embedding[idx]
            self.merge_lex(embed, 1)
        if pos in pos2id:
            self.pos[pos2id[pos]] += 1
        else:
            self.pos += 1/len(self.pos)
        if left_right == "l":
            self.left_right[0] += 1
        else:
            self.left_right[1] += 1
        if verb_voice == "a":
            self.verb_voice[0] += 1
        else:
            self.verb_voice[1] += 1
        self.relation[deprel2id[deprel]] += 1
        self.data.append(value)
    def __iter__(self):
        return self.data.__iter__()

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
                    if is_normal_argument(word_info[FLATTEN2ID[SR]]):
                        rela_position = 'r' if int(word_info[4]) > predicate_id else 'l'
                        verbvoice = 'p' if sentence[predicate_id-1][FLATTEN2ID[POS]] == 'VBN' else 'a'
                        deprel = word_info[FLATTEN2ID[DEPREL]]
                        preposition = _PAD_ if word_info[FLATTEN2ID[POS]] not in preposition2id else word_info[FLATTEN2ID[LEMMA]]
                        arg = word_info[FLATTEN2ID[SR]]
                        idx = (int(word_info[FLATTEN2ID[SENT_ID]]),int(word_info[FLATTEN2ID[PREDICATE_ID]]),int(word_info[FLATTEN2ID[WORD_ID]]))
                        arghead = find_argument_head(sentence, word_info)[FLATTEN2ID[LEMMA]].lower()
                        if is_number(arghead):
                            arghead = _NUM_
                        if arg not in groundtruths[predicate]:
                            groundtruths[predicate][arg] = []
                        groundtruths[predicate][arg].append(idx)
                        if (deprel,verbvoice,rela_position,preposition) not in predicts[predicate]:
                            predicts[predicate][(deprel,verbvoice,rela_position,preposition)] = Cluster([])
                        predicts[predicate][(deprel,verbvoice,rela_position,preposition)].append(idx,word_info[8],arghead,rela_position,verbvoice,deprel)
            sentence = []
            predicate = None
            predicate_id = -1
        else:
            if is_predicate(word_info[FLATTEN2ID[IS_PREDICATE]]) and is_verb(word_info[FLATTEN2ID[POS]]):
                predicate = word_info[FLATTEN2ID[LEMMA]]
                predicate_id = int(word_info[FLATTEN2ID[WORD_ID]])
                if predicate not in groundtruths:
                    groundtruths[predicate] = dict()
                    predicts[predicate] = dict()
            sentence.append(word_info)
    assert(len(sentence)==0 and predicate is None)
    return groundtruths, predicts

def merge_phases(predicts: Dict[str, Dict[Any, Any]], alpha: float) -> Dict[str, Dict[Any, Any]]:
    print("=== merging ===")

    no_zero_predicts = dict()
    for word in predicts.keys():
        no_zero_predicts[word] = []
        for key in predicts[word].keys():
            if len(predicts[word][key]) != 0:
                no_zero_predicts[word].append(predicts[word][key])
        no_zero_predicts[word].sort(key = len, reverse=True)

    for beta in tqdm(np.arange(0.95, 0.5, -0.1)):
        Cluster.BETA = beta
        for word in no_zero_predicts.keys():
            c_i, c_j = 0, 0
            while c_i < len(no_zero_predicts[word]):
                c_j, max_score = -1, 0
                for j in range(c_i):
                    score = no_zero_predicts[word][c_i].score(no_zero_predicts[word][j])
                    if score > max_score:
                        max_score = score
                        c_j = j
                if max_score > alpha:
                    no_zero_predicts[word][c_j] += no_zero_predicts[word][c_i]
                    del no_zero_predicts[word][c_i]
                    for c_k in range(c_j, 0, -1):
                        if len(no_zero_predicts[word][c_k]) > len(no_zero_predicts[word][c_k-1]):
                            no_zero_predicts[word][c_k], no_zero_predicts[word][c_k-1] = no_zero_predicts[word][c_k-1], no_zero_predicts[word][c_k]
                        else:
                            break
                else:
                    c_i += 1
    final_predicts = dict()
    for word, clusters_list in no_zero_predicts.items():
        clusters_dict = {i: cluster for i, cluster in enumerate(clusters_list)}
        final_predicts[word] = clusters_dict
    return final_predicts


def main():
    Cluster.GAMMA = 0.97
    ALPHA = 0.88
    truths, predicts = split_phase(flattened_train_data_path)
    pre, coll, f1 = evaluation(truths, predicts)
    print(pre, coll, f1)
    final_pre = merge_phases(predicts, ALPHA)
    pre, coll, f1 = evaluation(truths, final_pre)
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
