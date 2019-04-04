from tqdm import tqdm
from multiprocessing import Pool

def eval_verb_worker(args):
    verb, groundtruth, predict = args
    same_matrix = {truth_label:{predict_label:0 for predict_label in predict.keys() } for truth_label in groundtruth.keys()}
    groundtruth_same = {truth_label:0 for truth_label in groundtruth.keys()}
    predict_same = {predict_label:0 for predict_label in predict.keys()}
    instance2label = {value:key for key,values in groundtruth.items() for value in values}
    for predict_label, instances in predict.items():
        for instance in instances:
            truth_label = instance2label[instance]
            same_matrix[truth_label][predict_label] += 1
    for truth_label in same_matrix.keys():
        for predict_label, value in same_matrix[truth_label].items():
            groundtruth_same[truth_label] = max(groundtruth_same[truth_label], value)
            predict_same[predict_label] = max(predict_same[predict_label], value)
    precision = 0
    pre_base = 0
    collocation = 0
    coll_base = 0
    for key in groundtruth_same.keys():
        collocation += groundtruth_same[key]
        coll_base += len(groundtruth[key])
    for key in predict_same.keys():
        precision += predict_same[key]
        pre_base += len(predict[key])
    precision /= pre_base
    collocation /= coll_base
    assert(pre_base == coll_base)
    return verb, precision, collocation, coll_base


def fast_eval_f1(groundtruths, predicts):
    precisions = dict()
    collocations = dict()
    args = [(verb, groundtruths[verb], predicts[verb]) for verb in groundtruths.keys()]
    with Pool(10) as p:
      resultList = list(tqdm(p.imap(eval_verb_worker, args), total=len(args)))
    
    for verb, precision, collocation, base in resultList:
        precisions[verb] = (precision, base)
        collocations[verb] = (collocation, base)
    precision = 0
    collocation = 0
    base = 0
    for verb in groundtruths.keys():
        precision += precisions[verb][0]*precisions[verb][1]
        collocation += collocations[verb][0]*precisions[verb][1]
        base += precisions[verb][1]
    return precision/base, collocation/base, precisions, collocations





def eval_f1(groundtruths, predicts):
    precisions = dict()
    collocations = dict()
    
    for verb in tqdm(groundtruths.keys()):
        groundtruth = groundtruths[verb]
        predict = predicts[verb]
        groundtruth_same = {truth_label:0 for truth_label in groundtruth.keys()}
        predict_same = {predict_label:0 for predict_label in predict.keys()}
        for truth_label in groundtruth.keys():
            for predict_label in predict.keys():
                num = len(groundtruth[truth_label] & predict[predict_label])
                groundtruth_same[truth_label] = max(num, groundtruth_same[truth_label])
                predict_same[predict_label] = max(num, predict_same[predict_label])
        precisions[verb] = 0
        pre_base = 0
        collocations[verb] = 0
        coll_base = 0
        for key in groundtruth_same.keys():
            collocations[verb] += groundtruth_same[key]
            coll_base += len(groundtruth[key])
        for key in predict_same.keys():
            precisions[verb] += predict_same[key]
            pre_base += len(predict[key])
        precisions[verb] /= pre_base
        collocations[verb] /= coll_base
        
        precisions[verb] = (precisions[verb], pre_base)
        collocations[verb] = (collocations[verb], coll_base)
        assert(pre_base == coll_base)
    
    precision = 0
    collocation = 0
    base = 0
    for verb in groundtruths.keys():
        precision += precisions[verb][0]*precisions[verb][1]
        collocation += collocations[verb][0]*precisions[verb][1]
        base += precisions[verb][1]
    return precision/base, collocation/base, precisions, collocations

def eval_list(groundtruths, predicts):
    precisions = dict()
    collocations = dict()
    for verb in tqdm(groundtruths.keys()):
        groundtruth = groundtruths[verb]
        predict = predicts[verb]
        groundtruth_same = {truth_label:0 for truth_label in groundtruth.keys()}
        predict_same = {predict_label:0 for predict_label in predict.keys()}
        for truth_label in groundtruth.keys():
            for predict_label in predict.keys():
                i, j, same = 0, 0, 0
                while i < len(groundtruth[truth_label]) and j < len(predict[predict_label]):
                    if groundtruth[truth_label][i] == predict[predict_label][j]:
                        same += 1
                        i += 1
                        j += 1
                    elif groundtruth[truth_label][i] < predict[predict_label][j]:
                        i += 1
                    else:
                        j += 1
                groundtruth_same[truth_label] = max(same, groundtruth_same[truth_label])
                predict_same[predict_label] = max(same, predict_same[predict_label])
        precisions[verb] = 0
        pre_base = 0
        collocations[verb] = 0
        coll_base = 0
        for key in groundtruth_same.keys():
            collocations[verb] += groundtruth_same[key]
            coll_base += len(groundtruth[key])
        for key in predict_same.keys():
            precisions[verb] += predict_same[key]
            pre_base += len(predict[key])
        precisions[verb] /= pre_base
        collocations[verb] /= coll_base
        
        precisions[verb] = (precisions[verb], pre_base)
        collocations[verb] = (collocations[verb], coll_base)
        assert(pre_base == coll_base)
    
    precision = 0
    collocation = 0
    base = 0
    for verb in groundtruths.keys():
        precision += precisions[verb][0]*precisions[verb][1]
        collocation += collocations[verb][0]*precisions[verb][1]
        base += precisions[verb][1]
    return precision/base, collocation/base, precisions, collocations


def main():
    truths = {'is': {'A':{1,2,3,4}, 'B':{5,6,7,8,9}, 'C':{10,11,12}}}
    predicts = {'is': {'X':{1,2,3,5}, 'Y':{4}, 'Z':{6,7,8,9,10,11,12}}}
    pre, coll, _, _ = fast_eval_f1(truths, predicts)
    # 0.6666666666666666 0.8333333333333334
    print(pre, coll)

    truths = {
        'do': {'a':{i for i in range(10)}, 
                'b':{i for i in range(10,30)}, 
                'c':{i for i in range(30, 60)}, 
                'd':{i for i in range(60, 80)}, 
                'e':{i for i in range(80, 110)}, 
                'f':{i for i in range(110, 120)}, 
                'g':{i for i in range(120, 153)}},
        'love': {'a':{i+200 for i in range(5)}, 
                'b':{i+200 for i in range(5,10)}, 
                'c':{i+200 for i in range(10, 18)}, 
                'd':{i+200 for i in range(18, 26)}, 
                'e':{i+200 for i in range(26, 36)}, 
                'f':{i+200 for i in range(36, 46)}, 
                'g':{i+200 for i in range(46, 54)}},
        'hate': {'a':{i+400 for i in range(2)}, 
                'b':{i+400 for i in range(2,4)}, 
                'c':{i+400 for i in range(4, 6)}, 
                'd':{i+400 for i in range(6, 7)}, 
                'e':{i+400 for i in range(7, 9)}, 
                'f':{i+400 for i in range(9, 10)}, 
                'g':{i+400 for i in range(10, 20)}}
    }
    predicts = {
        'do': {'a':{i for i in range(10)} | {140}, 
                'b':{i for i in range(10,30)} | {141,142}, 
                'c':{i for i in range(30, 60)} | {143,144,145}, 
                'd':{i for i in range(60, 80)} | {146}, 
                'e':{i for i in range(80, 110)} | {147,148}, 
                'f':{i for i in range(110, 120)} | {149,150,151}, 
                'g':{i for i in range(120, 140)} | {152}},
        'love': {'a':{i+200 for i in range(5)} | {247}, 
                'b':{i+200 for i in range(5,10)} | {248}, 
                'c':{i+200 for i in range(10, 18)} | {249}, 
                'd':{i+200 for i in range(18, 26)} | {250}, 
                'e':{i+200 for i in range(26, 36)} | {251}, 
                'f':{i+200 for i in range(36, 46)} | {252}, 
                'g':{i+200 for i in range(46, 47)} | {253}},
        'hate': {'a':{i+400 for i in range(2)}, 
                'b':{i+400 for i in range(2,4)} | {411,412,413}, 
                'c':{i+400 for i in range(4, 6)} | {414}, 
                'd':{i+400 for i in range(6, 7)}, 
                'e':{i+400 for i in range(7, 9)} | {415,416}, 
                'f':{i+400 for i in range(9, 10)} | {417}, 
                'g':{i+400 for i in range(10, 11)} | {418,419}}
    }

    pre, coll, pres, colls = fast_eval_f1(truths, predicts)
    # 0.8942731277533039 0.8898678414096917 
    # {'do': (0.9215686274509803, 153), 'love': (0.8888888888888888, 54), 'hate': (0.7, 20)} 
    # {'do': (0.9215686274509803, 153), 'love': (0.8888888888888888, 54), 'hate': (0.65, 20)}
    print(pre, coll)
    print(pres)
    print(colls)

    truths_combine = {'combine': {'a':set(),'b':set(),'c':set(),'d':set(),'e':set(),'f':set(),'g':set()}}
    predicts_combine = {'combine': {'a':set(),'b':set(),'c':set(),'d':set(),'e':set(),'f':set(),'g':set()}}
    for word in truths:
        for key in truths[word].keys():
            truths_combine['combine'][key] |= truths[word][key]
            predicts_combine['combine'][key] |= predicts[word][key]
    
    pre, coll, _, _ = fast_eval_f1(truths_combine, predicts_combine)
    # 0.8898678414096917 0.8898678414096917
    print(pre, coll)

if __name__ == "__main__":
    main()
