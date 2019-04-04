import pickle
from tqdm import tqdm
from parameters import *
import time
from evaluation import evaluation
from copy import deepcopy

def exeTime(func):
    def newFunc(*args, **args2):
        t0 = time.time()
        print("@%s, {%s} start" % (time.strftime("%X", time.localtime()), func.__name__))
        back = func(*args, **args2)
        print("@%s, {%s} end" % (time.strftime("%X", time.localtime()), func.__name__))
        print("@%.3fs taken for {%s}" % (time.time() - t0, func.__name__))
        return back
    return newFunc

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


    sentence = []
    predicate = None
    for line in tqdm(sentences):
        word_info = line.strip().split()
        if len(word_info) == 0:
            if predicate is not None:
                for word_info in sentence:
                    if word_info[-1] != '_':
                        deprel = word_info[12]
                        arg = word_info[14]
                        idx = (word_info[0],word_info[1],word_info[4])
                        groundtruths[predicate][arg].append(idx)
                        predicts[predicate][deprel].append(idx)
            sentence = []
            predicate = None
        else:
            if word_info[3] == '1':
                predicate = word_info[6]
                if predicate not in groundtruths:
                    groundtruths[predicate] = deepcopy(init_arg_dict)
                    predicts[predicate] = deepcopy(init_deprel_dict)
            sentence.append(word_info)
    if len(sentence) != 0 and predicate is not None:
        for word_info in sentence:
            if word_info[-1] != '_':
                deprel = word_info[12]
                arg = word_info[14]
                idx = (word_info[0],word_info[1],word_info[4])
                groundtruths[predicate][arg].add(idx)
                predicts[predicate][deprel].add(idx)
    return groundtruths, predicts

@exeTime
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
