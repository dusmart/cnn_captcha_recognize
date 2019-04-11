import os
import pickle
import collections
import random
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Set, Callable
from parameters import *

_UP_ = 'up'
_DOWN_ = 'down'

INFOMATION = ['word_id', 'word', 'word_lemma_g', 'word_lemma_p', 'pos_g', 'pos_p', 'unk_1', 'unk_2', 'depid_g', 'depid_p', 
        'deprel_g', 'deprel_p', 'is_predicate']
ConllID = {info: idx for idx, info in enumerate(INFOMATION)}

class Path:
    def __init__(self, path_rels, path_directions, path_pos):
        self.path_rels = path_rels
        self.path_directions = path_directions
        self.path_pos = path_pos
def get_path_matrix(sentence):
    length = len(sentence)
    matrix = [[None] * length for _ in range(length)]
    for idx, word_info in enumerate(sentence):
        father_idx = int(word_info[ConllID["depid_g"]]) - 1
        deprel = word_info[ConllID["deprel_g"]]
        father_pos = sentence[father_idx][ConllID["pos_g"]]
        this_pos = word_info[ConllID["pos_g"]]
        if father_idx != -1:
            matrix[idx][father_idx] = Path([deprel], [_UP_], [this_pos])
            matrix[father_idx][idx] = Path([deprel], [_DOWN_], [father_pos])
    for k in range(length):
        for i in range(length):
            for j in range(length):
                if matrix[i][k] and matrix[k][j]:
                    path_rels = matrix[i][k].path_rels + matrix[k][j].path_rels
                    path_directions = matrix[i][k].path_directions + matrix[k][j].path_directions
                    path_pos = matrix[i][k].path_pos + matrix[k][j].path_pos
                    if matrix[i][j] is None:
                        matrix[i][j] = Path(path_rels, path_directions, path_pos)
                    elif len(matrix[i][j].path_rels) > len(path_rels):
                        matrix[i][j] = Path(path_rels, path_directions, path_pos)
    return matrix
    

def csv_string(string):
    return '"' + str(string) + '"'

def argument_csv_make(vocab_path, output_path):
    with open(vocab_path, 'r') as fin:
        data = fin.readlines()
    sentences = []
    sentence = []
    for i in range(len(data)):
        if len(data[i].strip())>0:
            sentence.append(data[i].strip().split('\t'))
        else:
            sentences.append(sentence)
            sentence = []
    if len(sentence) > 0:
        sentences.append(sentence)
    
    argument_infos = {}

    for sentence in tqdm(sentences):
        path_matrix = get_path_matrix(sentence)
        predicates_ids = []
        predicates_left_right_subj_idx = []
        subj_dix = None
        for idx, word_info in enumerate(sentence):
            if word_info[ConllID["pos_g"]] == "SBJ":
                subj_dix = idx
            if word_info[ConllID["is_predicate"]] == "Y":
                predicates_ids.append(idx)
                predicates_left_right_subj_idx.append(subj_dix)
        for predicates_no, predicates_id in enumerate(predicates_ids):
            if sentence[predicates_id][ConllID["pos_g"]][0] != "V":
                continue
            for idx, word_info in enumerate(sentence):
                if (idx == predicates_id):
                    continue

                pos = word_info[ConllID["pos_g"]]
                path = path_matrix[idx][predicates_id]

                father = int(word_info[ConllID["depid_g"]]) - 1
                direction_from_father_to_predicate = path_matrix[father][predicates_id].path_directions
                
                rule3 = predicates_left_right_subj_idx[predicates_no] == idx
                for direction in direction_from_father_to_predicate:
                    rule3 = rule3 and direction==_DOWN_


                path_direction = ""
                flag_down, flag_up = True, True
                for direction in path.path_directions:
                    if flag_down and direction == _DOWN_:
                        path_direction += direction
                        flag_down = False
                    if flag_up and direction == _UP_:
                        path_direction += direction
                        flag_up = False

                path_len = min(len(path.path_directions), 2)
                rule6 = is_predicates_child = len(path.path_directions) == 1
                rule7 = True
                for mid_pos in path.path_pos[1:]:
                    rule7 = rule7 and mid_pos[0] == "V"



                path_rel_end = path.path_rels[-1]
                is_argument = word_info[len(ConllID)+predicates_no] != "_"
                info = [pos, path_direction, path_len, path_rel_end, is_argument]
                info = map(csv_string, info)
                infostr = ",".join(info)
                if infostr in argument_infos:
                    argument_infos[infostr] += 1
                else:
                    argument_infos[infostr] = 1
    
    with open(output_path, 'w') as fout:
        #fout.write("pos, path_direction, path_len, path_rel_end, weight, is_argument\n")
        fout.write("""@relation argument_test_info

@attribute pos {NN,DT,NNP,IN,',',JJ,NNS,.,CD,VBD,RB,CC,VBZ,TO,PRP,VB,VBN,HYPH,POS,VBG,VBP,PRP$,MD,``,'\\'\\'',$,:,WDT,RP,RBR,WP,WRB,JJR,NNPS,),(,JJS,EX,RBS,PRF,PDT,WP$,#,UH,LS,NIL,SYM,FW}
@attribute 'path_direction' {updown,up,down}
@attribute 'path_len' {1, 2}
@attribute 'path_rel_end' {PMOD,OBJ,NMOD,VC,SBJ,ADV,IM,P,CONJ,APPO,OPRD,COORD,LOC,SUB,TMP,PRD,DIR,PRP,LGS,EXT,PRT,MNR,HMOD,AMOD,DEP,PRN,DTV,SUFFIX,PUT,EXTR,GAP-SBJ,NAME,HYPH,GAP-OBJ,EXT-GAP,BNF,GAP-VC,TITLE,LOC-OPRD,LOC-PRD,POSTHON,ADV-GAP,GAP-LOC,DEP-GAP,VOC,PRD-PRP,GAP-OPRD,GAP-PRD,GAP-MNR,AMOD-GAP,GAP-LGS,GAP-PUT,PRD-TMP,LOC-TMP,GAP-PMOD,GAP-NMOD,DIR-GAP,GAP-TMP,MNR-TMP,DIR-OPRD,LOC-MNR}
@attribute 'is_argument' {False,True}

@data
""")
        argument_infos = sorted(argument_infos.items(), key=lambda x:x[1], reverse=True)
        for argument_info in argument_infos:
            fout.write(argument_info[0]+",{"+str(argument_info[1])+"}\n")

rule1_dict = {"CC", "DT", "``", "$", ")", "(", ",", ".", "''", ":","#","SYM"}
rule2_dict = {("IM", _UP_+_DOWN_),("COORD", _UP_+_DOWN_), ("P", _UP_+_DOWN_), 
    ("DEP", _UP_+_DOWN_), ("SUB", _UP_+_DOWN_), ("PRT", _DOWN_), ("OBJ", _UP_),
    ("PMOD", _UP_), ("ADV", _UP_), ("ROOT", _UP_), ("TMP", _UP_), ("SBJ", _UP_), ("OPRD", _UP_)}

rule4_dict = {"ADV", "AMOD", "APPO", "BNF", "CONJ", "COORD", "DIR", "DTV", "EXT", "EXTR",
    "HMOD", "GAP-OBJ", "LGS", "LOC", "MNR", "NMOD", "OBJ", "OPRD", "POSTHON",
    "PRD", "PRN", "PRP", "PRT", "PUT", "SBJ", "SUB", "SUFFIX", "DEP"}
rule5_dict = {"be", "can", "could", "dare", "do", "have", "may", "might", "must", "need", "ought", "shall", "should", "will", "would"}

def split_merge_handrules(vocab_path, output_path):
    with open(vocab_path, 'r') as fin:
        data = fin.readlines()
    sentences = []
    sentence = []
    for i in range(len(data)):
        if len(data[i].strip())>0:
            sentence.append(data[i].strip().split('\t'))
        else:
            sentences.append(sentence)
            sentence = []
    if len(sentence) > 0:
        sentences.append(sentence)
    
    argument_infos = {}

    TrueLabelFalse = 0
    TrueLabelTrue = 0
    FalseLabelTrue = 0
    FalseLabelFalse = 0

    for sentence in tqdm(sentences):
        path_matrix = get_path_matrix(sentence)
        predicates_ids = []
        predicates_left_right_subj_idx = []
        subj_dix = -1
        for idx, word_info in enumerate(sentence):
            if word_info[ConllID["pos_g"]] == "SBJ":
                subj_dix = idx
            if word_info[ConllID["is_predicate"]] == "Y":
                predicates_ids.append(idx)
                predicates_left_right_subj_idx.append(subj_dix)
        for predicates_no, predicates_id in enumerate(predicates_ids):
            if sentence[predicates_id][ConllID["pos_g"]][0] != "V":
                continue
            for idx, word_info in enumerate(sentence):
                if (idx == predicates_id):
                    continue

                lemma = word_info[ConllID["word_lemma_g"]]
                pos = word_info[ConllID["pos_g"]]
                path = path_matrix[predicates_id][idx]
                father = int(word_info[ConllID["depid_g"]]) - 1
                direction_from_father_to_predicate = path_matrix[father][predicates_id].path_directions
                
                path_direction = ""
                flag_down, flag_up = True, True
                for direction in path.path_directions:
                    if flag_down and direction == _DOWN_:
                        path_direction += direction
                        flag_down = False
                    if flag_up and direction == _UP_:
                        path_direction += direction
                        flag_up = False

                rule1 = pos in rule1_dict
                rule2 = (path.path_rels[-1], path_direction) in rule2_dict
                rule3 = predicates_left_right_subj_idx[predicates_no] == idx
                for direction in direction_from_father_to_predicate:
                    rule3 = rule3 and direction==_DOWN_
                rule3 = not rule3
                rule4 = False
                for rel in path.path_rels[:-1]:
                    if rel in rule4_dict:
                        rule4 = True
                        break
                rule5 = lemma in rule5_dict and idx+1<len(sentence) and sentence[idx+1][ConllID["pos_g"]][0]=="V"
                rule5 = not rule5

                rule6 =  father == predicates_id
                rule7 = True
                for mid_pos in path.path_pos[1:]:
                    rule7 = rule7 and mid_pos[0] == "V"

                rule8 = None
                if rule8 is None and rule1: rule8 = False
                if rule8 is None and rule2: rule8 = False
                if rule8 is None and rule3: rule8 = True
                if rule8 is None and rule4: rule8 = False
                if rule8 is None and rule5: rule8 = False
                if rule8 is None and rule6: rule8 = True
                if rule8 is None and rule7: rule8 = True
                if rule8 is None: rule8 = False


                is_argument = word_info[len(ConllID)+predicates_no] != "_"
                if rule8 and is_argument:
                    TrueLabelTrue += 1
                elif rule8 and not is_argument:
                    FalseLabelTrue += 1
                elif not rule8 and is_argument:
                    TrueLabelFalse += 1
                else:
                    FalseLabelFalse += 1

                info = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, is_argument]
                info = map(csv_string, info)
                infostr = ",".join(info)
                if infostr in argument_infos:
                    argument_infos[infostr] += 1
                else:
                    argument_infos[infostr] = 1
    precision_true = TrueLabelTrue / (TrueLabelTrue+FalseLabelTrue)
    recall_true = TrueLabelTrue / (TrueLabelTrue+TrueLabelFalse)
    f1_true = 2*precision_true*recall_true/(precision_true+recall_true)
    precision_false = FalseLabelFalse / (FalseLabelFalse+TrueLabelFalse)
    recall_false = FalseLabelFalse / (FalseLabelFalse+FalseLabelTrue)
    f1_false = 2*precision_false*recall_false/(precision_false+recall_false)
    print(precision_true, recall_true, f1_true)
    print(precision_false, recall_false, f1_false)
    with open(output_path, 'w') as fout:
        #fout.write("pos, path_direction, path_len, path_rel_end, weight, is_argument\n")
        fout.write("""@relation argument_test_info

@attribute 'rule1' {True, False}
@attribute 'rule2' {True, False}
@attribute 'rule3' {True, False}
@attribute 'rule4' {True, False}
@attribute 'rule5' {True, False}
@attribute 'rule6' {True, False}
@attribute 'rule7' {True, False}
@attribute 'rule8' {True, False}
@attribute 'is_argument' {False, True}
@data
""")
        argument_infos = sorted(argument_infos.items(), key=lambda x:x[1], reverse=True)
        for argument_info in argument_infos:
            fout.write(argument_info[0]+",{"+str(argument_info[1])+"}\n")



def test():
    pass


def main():
    #argument_csv_make(raw_train_data_path, "argument_train_info.arff")
    split_merge_handrules(raw_test_data_path, "train_arff/argument_test_info.arff")

if __name__ == "__main__":
    main()