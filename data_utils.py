import os
import time
import pickle
import collections
import random
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Set, Callable
from parameters import *

_UNK_ = '<UNK>'
_PAD_ = '<PAD>'
_ROOT_ = '<ROOT>'
_NUM_ = '<NUM>'

def print_exe_time(func):
    def wapper(*args, **kwargs):
        tic = time.time()
        # print("@%s, {%s} start" % (time.strftime("%X", time.localtime()), func.__name__))
        value = func(*args, **kwargs)
        tac = time.time()
        # print("@%s, {%s} end" % (time.strftime("%X", time.localtime()), func.__name__))
        print("===== @%.2fs taken for {%s} =====" % (tac - tic, func.__name__))
        return value
    return wapper

NOT_ARGUMENT_HEAD = {"IN", "DT", "TO"}

def find_argument_head(sentence: List[List[str]], word_info: List[str]) -> List[str]:
    """ find an argument's nominal head
    """
    pos_tag = word_info[CONLL2ID[POS]]
    if pos_tag != "IN":
        return word_info
    for i in range(len(sentence)-1, -1, -1):
        if sentence[i][CONLL2ID[DEPHEAD]] == word_info[CONLL2ID[WORD_ID]]:
            return find_argument_head(sentence, sentence[i])
    return word_info

def find_argument_span(sentence: List[List[str]], word_info: List[str]) -> str:
    result = "|" + word_info[CONLL2ID[LEMMA]].lower() + "|"
    head = {word_info[CONLL2ID[WORD_ID]]}
    start = int(word_info[CONLL2ID[WORD_ID]])-1
    end = int(word_info[CONLL2ID[WORD_ID]])-1
    for i in range(end+1, len(sentence)):
        if i < len(sentence) and sentence[i][CONLL2ID[DEPHEAD]] in head:
            end += 1
            result += " " + sentence[i][CONLL2ID[LEMMA]].lower()
            head.add(sentence[i][CONLL2ID[WORD_ID]])
        else:
            break
    for i in range(start-1, -1, -1):
        if i >= 0 and sentence[i][CONLL2ID[DEPHEAD]] in head:
            start -= 1
            result = sentence[i][CONLL2ID[LEMMA]].lower() + " " + result
            head.add(sentence[i][CONLL2ID[WORD_ID]])
        else:
            break
    return result

def is_scientific_notation(s: str) -> bool:
    s = str(s)
    if s.count(',')>=1:
        sl = s.split(',')
        for item in sl:
            if not item.isdigit():
                return False
        return True   
    return False

def is_float(s: str) -> bool:
    s = str(s)
    if s.count('.')==1:
        sl = s.split('.')
        left = sl[0]
        right = sl[1]
        if left.startswith('-') and left.count('-')==1 and right.isdigit():
            lleft = left.split('-')[1]
            if lleft.isdigit() or is_scientific_notation(lleft):
                return True
        elif (left.isdigit() or is_scientific_notation(left)) and right.isdigit():
            return True
    return False

def is_fraction(s: str) -> bool:
    s = str(s)
    if s.count('\\/')==1:
        sl = s.split('\\/')
        if len(sl)== 2 and sl[0].isdigit() and sl[1].isdigit():
            return True  
    if s.count('/')==1:
        sl = s.split('/')
        if len(sl)== 2 and sl[0].isdigit() and sl[1].isdigit():
            return True    
    if s[-1]=='%' and len(s)>1:
        return True
    return False

def is_number(s: str) -> bool:
    s = str(s)
    if s.isdigit() or is_float(s) or is_fraction(s) or is_scientific_notation(s):
        return True
    else:
        return False

class VocabMaker:
    """ vocabulary maker for different kind of tags from conll09 data
        vocabulary extraction rule in defined in `get_filter_funcs`
        default word for some special word will be added by `get_paddings`
    """
    def __init__(self, train_data_path: str, test_data_path: str, dev_data_path: str) -> None:
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.dev_data_path = dev_data_path
    def make_vocab(self, vocab_path: str, symbol2idx_path: str, idx2symbol_path: str, symbol_type: str, \
                    use_lower_bound: bool = False, freq_lower_bound: int = 0, quiet: bool = False) -> None:
        """ parse Conll09 data file, make a vocabulary and store it to given path
        filter_func: filter specific word from a sentence and the specific word_line
        """
        # 0. get filter and paddings by symbol type
        padding_symbols = self.get_paddings(symbol_type)
        filter_func = self.get_filter_funcs(symbol_type)
        # 1. read sentences
        with open(self.train_data_path,'r') as fin:
            data = fin.readlines()
        with open(self.test_data_path, 'r') as fin:
            data += fin.readlines()
        with open(self.dev_data_path, 'r') as fin:
            data += fin.readlines()
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
        # 2. add symbols into list
        symbol_data = []
        for sentence in tqdm(sentences):
            for word_line in sentence:
                filter_func(symbol_data, sentence, word_line)
        # 3. count add make vocabulary frequency dict
        symbol_data_counter = collections.Counter(symbol_data).most_common()
        if use_lower_bound:
            symbol_vocab = padding_symbols + [item[0] for item in symbol_data_counter if item[1]>=freq_lower_bound]
        else:
            symbol_vocab = padding_symbols + [item[0] for item in symbol_data_counter]
        symbol_to_idx = {word:idx for idx,word in enumerate(symbol_vocab)}
        idx_to_symbol = {idx:word for idx,word in enumerate(symbol_vocab)}
        # 4. print infomation
        if not quiet:
            print('\t{} vocab size:{}'.format(symbol_type, len(symbol_vocab)))
            print('\tdump vocab at:{}'.format(vocab_path))
        # 5. save to file
        with open(vocab_path, 'w') as f:
            for item in symbol_data_counter:
                f.write(str(item[0]) + "\t" + str(item[1]) + "\n")
        with open(symbol2idx_path,'wb') as f:
            pickle.dump(symbol_to_idx,f)
        with open(idx2symbol_path,'wb') as f:
            pickle.dump(idx_to_symbol,f)

    def get_filter_funcs(self, symbol_type):
        # to reduce data sparsity, we only add lower non-numeric words
        if symbol_type == WORD or symbol_type == LEMMA:
            def filter_func(symbol_data, sentence, word_line):
                word_lower = word_line[CONLL2ID[symbol_type]].lower()
                if not is_number(word_lower):
                    symbol_data.append(word_lower)
            return filter_func
        # preposition is recognized by there pos tag
        elif symbol_type == "preposition":
            def filter_func(symbol_data, sentence, word_line):
                if is_preposition(word_line[CONLL2ID[POS]]):
                    word_lower = word_line[CONLL2ID[LEMMA]].lower()
                    if not is_number(word_lower):
                        symbol_data.append(word_lower)
            return filter_func
        # directly filter the pos tag or dependency relation from data line
        elif symbol_type == POS or symbol_type == DEPREL:
            def filter_func(symbol_data, sentence, word_line):
                symbol_data.append(word_line[CONLL2ID[symbol_type]])
            return filter_func
        # argument label are lied after those normal tags
        elif symbol_type == "arg":
            def filter_func(symbol_data, sentence, word_line):
                for i in range(len(CONLL2ID), len(word_line)):
                    symbol_data.append(word_line[i])
            return filter_func
        # we only interested in normal argument's head information, it's useful head or whole span
        elif symbol_type == "arghead" or symbol_type == "argspan":
            def filter_func(symbol_data, sentence, word_line):
                flag = 0
                for i in range(len(CONLL2ID), len(word_line)):
                    if is_normal_argument(word_line[i]):
                        flag += 1
                if flag:
                    if symbol_type == "arghead":
                        lemma = find_argument_head(sentence, word_line)[CONLL2ID[LEMMA]]
                    elif symbol_type == "argspan":
                        lemma = find_argument_span(sentence, word_line)
                    lemma = lemma.lower()
                    if not is_number(lemma):
                        for _ in range(flag):
                            symbol_data.append(lemma)
            return filter_func
        # we only interested in normal argument's pos tag
        elif symbol_type == "arg_pos_rel":
            def filter_func(symbol_data, sentence, word_line):
                for i in range(len(CONLL2ID), len(word_line)):
                    if is_normal_argument(word_line[i]):
                        symbol_data.append(word_line[CONLL2ID[POS]])
            return filter_func
        # we don't support other symbols, add it if you need
        else:
            raise Exception("symbol type {} not supported now".format(symbol_type))

    def get_paddings(self, symbol_type):
        if symbol_type == WORD or symbol_type == LEMMA:
            return [_PAD_, _UNK_, _ROOT_, _NUM_]
        elif symbol_type == "preposition":
            return [_PAD_, _UNK_]
        elif symbol_type == POS:
            return [_PAD_,_UNK_,_ROOT_]
        elif symbol_type == DEPREL:
            return [_PAD_,_UNK_]
        elif symbol_type == "arg":
            return [_PAD_,_UNK_]
        elif symbol_type == "arghead":
            return [_PAD_,_UNK_, _NUM_]
        else:
            return []

def shrink_pretrained_embedding(train_file, dev_file, test_file, pretrained_file, pretrained_emb_size, pretrained_embed_path, id2pretrained_path, pretrained2id_path, pretrained_vocab_path, quiet=False):
    """ shrink the embedding file to only those words occured in our dataset
    """
    word_set = set()
    with open(train_file,'r') as f:
        data = f.readlines()
        for line in data:
            if len(line.strip())>0:
                line = line.strip().split('\t')
                word_set.add(line[CONLL2ID[WORD]].lower())
                word_set.add(line[CONLL2ID[LEMMA]].lower())
    with open(dev_file,'r') as f:
        data = f.readlines()
        for line in data:
            if len(line.strip())>0:
                line = line.strip().split('\t')
                word_set.add(line[CONLL2ID[WORD]].lower())
                word_set.add(line[CONLL2ID[LEMMA]].lower())
    with open(test_file,'r') as f:
        data = f.readlines()
        for line in data:
            if len(line.strip())>0:
                line = line.strip().split('\t')
                word_set.add(line[CONLL2ID[WORD]].lower())
                word_set.add(line[CONLL2ID[LEMMA]].lower())

    pretrained_vocab = [_PAD_,_UNK_,_ROOT_,_NUM_]
    pretrained_embedding = [
                            [0.0]*pretrained_emb_size,
                            [0.0]*pretrained_emb_size,
                            [0.0]*pretrained_emb_size,
                            [0.0]*pretrained_emb_size
                        ]

    with open(pretrained_file,'r') as f:
        for line in f.readlines():
            row = line.split(' ')
            word = row[0].lower()
            if word in word_set:
                pretrained_vocab.append(word)
                weight = [float(item) for item in row[1:]]
                assert(len(weight)==pretrained_emb_size)
                pretrained_embedding.append(weight)

    pretrained_embedding = np.array(pretrained_embedding,dtype=float)

    pretrained_to_idx = {word:idx for idx,word in enumerate(pretrained_vocab)}

    idx_to_pretrained = {idx:word for idx,word in enumerate(pretrained_vocab)}

    if not quiet:
        print('\tshrink pretrained vocab size:{}'.format(len(pretrained_vocab)))
        print('\tdataset sum:{} pretrained cover:{} coverage:{:.3}%'.format(len(word_set),len(pretrained_vocab),len(pretrained_vocab)/len(word_set)*100))

    if not quiet:
        print('\tdump vocab at:{}'.format(pretrained_vocab_path))

    with open(pretrained_vocab_path, 'w') as f:
        f.write('\n'.join(pretrained_vocab))

    with open(pretrained2id_path,'wb') as f:
        pickle.dump(pretrained_to_idx,f)

    with open(id2pretrained_path,'wb') as f:
        pickle.dump(idx_to_pretrained,f)

    with open(pretrained_embed_path,'wb') as f:
        pickle.dump(pretrained_embedding,f)

# be careful when changing numbers in this function, I used raw number directly
def flat_dataset(dataset_file, output_path):
    """ flatten a conll09 data file to flattened data file, see README for format details
    """
    with open(dataset_file,'r') as f:
        data = f.readlines()
    origin_data = []
    sentence = []
    for i in range(len(data)):
        if len(data[i].strip())>0:
            sentence.append(data[i].strip().split('\t'))
        else:
            origin_data.append(sentence)
            sentence = []
    if len(sentence) > 0:
        origin_data.append(sentence)

    with open(output_path, 'w') as f:
        for sidx in tqdm(range(len(origin_data))):
            sentence = origin_data[sidx]
            # change those number into _NUM_
            for i in range(len(sentence)):
                word_info = sentence[i]
                for j in range(1, 4):
                    if is_number(word_info[j].lower()):
                        word_info[j] = _NUM_
            predicate_idx = 0
            for i in range(len(sentence)):
                if sentence[i][CONLL2ID[IS_PREDICATE]] == 'Y':
                    output_block = []
                    for j in range(len(sentence)):
                        word_info = sentence[j]
                        IS_PRED = int(i == j)
                        tag = sentence[j][14+predicate_idx]
                        output_block.append([str(sidx), str(predicate_idx), str(len(sentence)), str(IS_PRED)]+word_info[0:1]+word_info[1:6]+word_info[8:12]+[tag])
                    for item in output_block:
                        f.write('\t'.join(item))
                        f.write('\n')
                    f.write('\n')
                    predicate_idx += 1
