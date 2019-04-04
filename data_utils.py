import os
import pickle
import collections
import random
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Set, Callable

_UNK_ = '<UNK>'
_PAD_ = '<PAD>'
_ROOT_ = '<ROOT>'
_NUM_ = '<NUM>'

class Vertex:
    """ vertex for dependency tree
    """
    def __init__(self, id, head):
        self.id = id
        self.head = head
        self.children = []

def is_valid_tree(sentence: List[List[str]], rd_node: int, cur_node: int) -> bool:
    """ judge if cur_node is not in the path from rd_node to root
    """
    if rd_node == 0:
        return True
    if rd_node == cur_node:
        return False
    cur_head = int(sentence[rd_node-1][9])
    if cur_head == cur_node:
        return False
    while cur_head != 0:
        cur_head = int(sentence[cur_head-1][9])
        if cur_head == cur_node:
            return False
    return True


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
    def __init__(self):
        pass
    def make_vocab(self, file_name: str, vocab_path: str, symbol2idx_path: str, idx2symbol_path: str, symbol_type: str, \
                    use_lower_bound: bool = False, freq_lower_bound: int = 0, quiet: bool = False) -> None:
        """ parse Conll09 data file, make a vocabulary and store it to given path
        filter_func: filt specific word from a sentence and the specific word_line
        """
        # 0. get filter and paddings by symbol type
        padding_symbols = self.get_paddings(symbol_type)
        filter_func = self.get_filter_funcs(symbol_type)
        # 1. read sentences
        with open(file_name,'r') as f:
            data = f.readlines()
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
            f.write('\n'.join(vocab_path))
        with open(symbol2idx_path,'wb') as f:
            pickle.dump(symbol_to_idx,f)
        with open(idx2symbol_path,'wb') as f:
            pickle.dump(idx_to_symbol,f)

    def get_filter_funcs(self, symbol_type):
        if symbol_type == "word":
            def filter_func(symbol_data, sentence, word_line):
                if not is_number(word_line[1].lower()):
                    symbol_data.append(word_line[1].lower())
            return filter_func
        elif symbol_type == "lemma":
            def filter_func(symbol_data, sentence, word_line):
                if not is_number(word_line[3].lower()):
                    symbol_data.append(word_line[3].lower())
            return filter_func
        elif symbol_type == "preposition":
            def filter_func(symbol_data, sentence, word_line):
                if word_line[4] == 'IN':
                    symbol_data.append(word_line[2])
            return filter_func
        elif symbol_type == "pos":
            def filter_func(symbol_data, sentence, word_line):
                symbol_data.append(word_line[5])
            return filter_func
        elif symbol_type == "deprel":
            def filter_func(symbol_data, sentence, word_line):
                symbol_data.append(word_line[10])
            return filter_func
        elif symbol_type == "arg":
            def filter_func(symbol_data, sentence, word_line):
                for i in range(len(word_line)-14):
                    symbol_data.append(word_line[14+i])
            return filter_func
        elif symbol_type == "arghead":
            def filter_func(symbol_data, sentence, word_line):
                for i in range(len(word_line)-14):
                    if word_line[14+i] != '_':
                        symbol_data.append(sentence[int(word_line[8])-1][2])
            return filter_func
        else:
            raise Exception("symbol type {} not supported now".format(symbol_type))

    def get_paddings(self, symbol_type):
        if symbol_type == "word" or symbol_type == "lemma":
            return [_PAD_, _UNK_, _ROOT_, _NUM_]
        elif symbol_type == "preposition":
            return [_PAD_, _UNK_]
        elif symbol_type == "pos":
            return [_PAD_,_UNK_,_ROOT_]
        elif symbol_type == "deprel":
            return [_PAD_,_UNK_]
        elif symbol_type == "arg":
            return [_PAD_,_UNK_]
        elif symbol_type == "arghead":
            return [_PAD_,_UNK_]
        else:
            raise Exception("symbol type {} not supported now".format(symbol_type))



def count_sentence_predicate(sentence):
    count = 0
    for item in sentence:
        if item[12] == 'Y':
            assert item[12] == 'Y' and item[13] != '_'
            count += 1
    return count

def shrink_pretrained_embedding(train_file, dev_file, test_file, pretrained_file, pretrained_emb_size, output_path, quiet=False):
    word_set = set()
    with open(train_file,'r') as f:
        data = f.readlines()
        for line in data:
            if len(line.strip())>0:
                line = line.strip().split('\t')
                word_set.add(line[1].lower())
    with open(dev_file,'r') as f:
        data = f.readlines()
        for line in data:
            if len(line.strip())>0:
                line = line.strip().split('\t')
                word_set.add(line[1].lower())

    with open(test_file,'r') as f:
        data = f.readlines()
        for line in data:
            if len(line.strip())>0:
                line = line.strip().split('\t')
                word_set.add(line[1].lower())

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
        print('\tdump vocab at:{}'.format(output_path))

    vocab_path = os.path.join(output_path,'pretrain.vocab')

    pretrain2idx_path = os.path.join(output_path,'pretrain2idx.bin')

    idx2pretrain_path = os.path.join(output_path,'idx2pretrain.bin')

    pretrain_emb_path = os.path.join(output_path,'pretrain.emb.bin')

    with open(vocab_path, 'w') as f:
        f.write('\n'.join(pretrained_vocab))

    with open(pretrain2idx_path,'wb') as f:
        pickle.dump(pretrained_to_idx,f)

    with open(idx2pretrain_path,'wb') as f:
        pickle.dump(idx_to_pretrained,f)

    with open(pretrain_emb_path,'wb') as f:
        pickle.dump(pretrained_embedding,f)


def flat_dataset(dataset_file, output_path):
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
                for _ in range(1, 4):
                    if is_number(word_info[1].lower()):
                        word_info[1] = _NUM_

            predicate_idx = 0
            for i in range(len(sentence)):
                if sentence[i][12] == 'Y':
                    output_block = []
                    for j in range(len(sentence)):
                        word_info = sentence[j]
                        IS_PRED = int(i == j)
                        tag = sentence[j][14+predicate_idx] # APRED
                        output_block.append([str(sidx), str(predicate_idx), str(len(sentence)), str(IS_PRED)]+word_info[0:1]+word_info[1:6]+word_info[8:12]+[tag])
                    
                    for item in output_block:
                        f.write('\t'.join(item))
                        f.write('\n')
                    f.write('\n')
                    predicate_idx += 1


def stat_max_order(dataset_file):
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

    max_order = 0

    for sidx in tqdm(range(len(origin_data))):
        sentence = origin_data[sidx]
        predicate_idx = 0

        for i in range(len(sentence)):
            if sentence[i][12] == 'Y':
                
                argument_set = set()
                for j in range(len(sentence)):
                    if sentence[j][14+predicate_idx] != '_':
                        argument_set.add(int(sentence[j][0]))
                
                cur_order = 1
                while True:
                    found_set = set()
                    son_data = []
                    order_idx = 0
                    while order_idx < cur_order:
                        son_order = [[] for _ in range(len(sentence)+1)]
                        for j in range(len(sentence)):
                            if len(son_data) == 0:
                                son_order[int(sentence[j][9])].append(int(sentence[j][0]))
                            else:
                                for k in range(len(son_data[-1])):
                                    if int(sentence[j][9]) in son_data[-1][k]:
                                        son_order[k].append(int(sentence[j][0]))
                                        break
                        son_data.append(son_order)
                        order_idx += 1
                    
                    current_node = int(sentence[i][0])
                    while True:
                        for item in son_data:
                            found_set.update(item[current_node])
                        if current_node != 0:
                            current_node = int(sentence[current_node-1][9])
                        else:
                            break
                    if len(argument_set - found_set) > 0:
                        cur_order += 1
                    else:
                        break
                if cur_order > max_order:
                    max_order = cur_order
                predicate_idx += 1

    print('max order:{}'.format(max_order))




def load_dataset_input(file_path):
    with open(file_path,'r') as f:
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

    return origin_data

def load_deprel_vocab(path):
    with open(path,'r') as f:
        data = f.readlines()
    
    data = [item.strip() for item in data if len(item.strip())>0 and item.strip()!=_UNK_ and item.strip()!=_PAD_]

    return data

def output_predict(path, data):
    with open(path, 'w') as f:
        for sentence in data:
            for i in range(len(sentence[0])):
                line = [str(sentence[j][i]) for j in range(len(sentence))]
                f.write('\t'.join(line))
                f.write('\n')
            f.write('\n')
