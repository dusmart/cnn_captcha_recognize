import pickle
from tqdm import tqdm
from parameters import *
import time

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


init_deprel_dict = {deprel:set() for deprel in deprel2id.keys()}
init_arg_dict = {arg:set() for arg in arg2id.keys()}

    


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
                        groundtruths[predicate][arg].add(idx)
                        predicts[predicate][deprel].add(idx)
            sentence = []
            predicate = None
        else:
            if word_info[3] == '1' and 'V' == word_info[8][0]:
                predicate = word_info[6]
                if predicate not in groundtruths:
                    groundtruths[predicate] = init_arg_dict.copy()
                    predicts[predicate] = init_deprel_dict.copy()
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
    truths, predicts = get_label(flattened_train_data_path)
    from evaluation import fast_eval_f1, eval_f1
    pre, coll, _, _ = fast_eval_f1(truths, predicts)
    print(pre, coll)

    truths_combine = {'combine': init_arg_dict.copy()}
    predicts_combine = {'combine': init_deprel_dict.copy()}
    for word in truths:
        for key in truths[word].keys():
            truths_combine['combine'][key] = truths_combine['combine'][key] | truths[word][key]
    for word in predicts:
        for key in predicts[word].keys():
            predicts_combine['combine'][key] = predicts_combine['combine'][key] | predicts[word][key]
    
    pre, coll, _, _ = fast_eval_f1(truths_combine, predicts_combine)
    print(pre, coll)

if __name__ == "__main__":
    main()
