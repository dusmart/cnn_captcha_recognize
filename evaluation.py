from tqdm import tqdm
from typing import Dict, List, Tuple, Any

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

def evaluation(groundtruths: Dict[str, Dict[Any, List[Any]]], 
                predicts: Dict[str, Dict[Any, List[Any]]]) -> Tuple[float]:
    precisions = dict()
    collocations = dict()
    print("=== evaluating on clusters ====")
    for verb in tqdm(groundtruths.keys()):
        groundtruth = groundtruths[verb]
        predict = predicts[verb]
        groundtruth_same = {truth_label:0 for truth_label in groundtruth.keys()}
        predict_same = {predict_label:0 for predict_label in predict.keys()}
        same_matrix = {truth_label:{predict_label:0 for predict_label in predict.keys() } for truth_label in groundtruth.keys()}
        instance2label = {value:key for key,values in groundtruth.items() for value in values}
        
        for predict_label, instances in predict.items():
            for instance in instances:
                truth_label = instance2label[instance]
                same_matrix[truth_label][predict_label] += 1
        for truth_label in same_matrix.keys():
            for predict_label, value in same_matrix[truth_label].items():
                groundtruth_same[truth_label] = max(groundtruth_same[truth_label], value)
                predict_same[predict_label] = max(predict_same[predict_label], value)

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
    avg_pre, avg_coll = precision/base, collocation/base
    return avg_pre, avg_coll, 2*avg_coll*avg_pre/(avg_coll+avg_pre)

def main():
    truths = {'is': {'A':[1,2,3,4], 'B':[5,6,7,8,9], 'C':[10,11,12]}}
    predicts = {'is': {'X':[1,2,3,5], 'Y':[4], 'Z':[6,7,8,9,10,11,12]}}
    pre, coll, f1 = evaluation(truths, predicts)
    # 0.6666666666666666 0.8333333333333334 0.7407407407407408
    print(pre, coll, f1)

    truths = {
        'do': {'a':[i for i in range(10)],
                'b':[i for i in range(10,30)], 
                'c':[i for i in range(30, 60)], 
                'd':[i for i in range(60, 80)], 
                'e':[i for i in range(80, 110)], 
                'f':[i for i in range(110, 120)], 
                'g':[i for i in range(120, 153)]},
        'love': {'a':[i+200 for i in range(5)],
                'b':[i+200 for i in range(5,10)], 
                'c':[i+200 for i in range(10, 18)], 
                'd':[i+200 for i in range(18, 26)], 
                'e':[i+200 for i in range(26, 36)], 
                'f':[i+200 for i in range(36, 46)], 
                'g':[i+200 for i in range(46, 54)]},
        'hate': {'a':[i+400 for i in range(2)],
                'b':[i+400 for i in range(2,4)], 
                'c':[i+400 for i in range(4, 6)], 
                'd':[i+400 for i in range(6, 7)], 
                'e':[i+400 for i in range(7, 9)], 
                'f':[i+400 for i in range(9, 10)], 
                'g':[i+400 for i in range(10, 20)]}
    }
    predicts = {
        'do': {'a':[i for i in range(10)] + [140], 
                'b':[i for i in range(10,30)] + [141,142], 
                'c':[i for i in range(30, 60)] + [143,144,145], 
                'd':[i for i in range(60, 80)] + [146], 
                'e':[i for i in range(80, 110)] + [147,148], 
                'f':[i for i in range(110, 120)] + [149,150,151], 
                'g':[i for i in range(120, 140)] + [152]},
        'love': {'a':[i+200 for i in range(5)] + [247], 
                'b':[i+200 for i in range(5,10)] + [248], 
                'c':[i+200 for i in range(10, 18)] + [249], 
                'd':[i+200 for i in range(18, 26)] + [250], 
                'e':[i+200 for i in range(26, 36)] + [251], 
                'f':[i+200 for i in range(36, 46)] + [252], 
                'g':[i+200 for i in range(46, 47)] + [253]},
        'hate': {'a':[i+400 for i in range(2)], 
                'b':[i+400 for i in range(2,4)] + [411,412,413], 
                'c':[i+400 for i in range(4, 6)] + [414], 
                'd':[i+400 for i in range(6, 7)], 
                'e':[i+400 for i in range(7, 9)] + [415,416], 
                'f':[i+400 for i in range(9, 10)] + [417], 
                'g':[i+400 for i in range(10, 11)] + [418,419]}
    }

    pre, coll, f1 = evaluation(truths, predicts)
    # 0.8942731277533039 0.8898678414096917 0.8920650459563823
    print(pre, coll, f1)

    truths_combine = {'combine': {'a':[],'b':[],'c':[],'d':[],'e':[],'f':[],'g':[]}}
    predicts_combine = {'combine': {'a':[],'b':[],'c':[],'d':[],'e':[],'f':[],'g':[]}}
    for word in truths:
        for key in truths[word].keys():
            truths_combine['combine'][key] += truths[word][key]
            predicts_combine['combine'][key] += predicts[word][key]
    
    pre, coll, f1 = evaluation(truths_combine, predicts_combine)
    # 0.8898678414096917 0.8898678414096917 0.8898678414096917
    print(pre, coll, f1)

if __name__ == "__main__":
    main()
