# a word-embedding similarity driven split-merge method for unsupervised semantic role labeling(SRL)

## 1. What does this repo do?

I implemented an **unsupervised SRL** method described in **Unsupervised Semantic Role Induction via Split-Merge Clustering**. After that, I used Glove embedding of arguments for impoving the effect of merging phase.

I used Conll09's training data for **evaluating** the method whose format will be shown in next sections. 

## 2. How to use the code?

1. clone this repo
2. download Glove_100d to folder pretrained_model named as glove.100d.txt
3. download conll09 data to raw_data folder and named as conll09_{train,dev,test}.dataset
4. install some required python packages if you failed running codes below
5. run `python prepare.py` which will flatten all data and shrink the word embedding size
6. run `python baseline.py` to get the baseline results
7. run `python split_merge.py` to get the algorithm described in paper above

## 3. floder structure

1. flattened_data/:     flattened conll09 data (duplicate sentence for its every predicate)
2. labels/:             vocabulary built for tags and some other information
3. pretrained_model/:   word's embedding collected from Glove(100d) and shrinked data
4. raw_data/:           Conll09 data
5. temp/
6. trash/
7. parameters.py:       some folder and file names and the data's label-id converter
8. data_utils.py:       some helper functions and classes
9. prepare.py:          for flattening data and shrink pretrained word's embedding
10. evaluation.py:      for evaluating the cluster effect, will return purity&collocation&F1
11. baseline.py:        assign argument's dependency relation as a their role, for baseline experiments
12. split_merge.py:     re-implement the algorithm decribed in previous paper
13. glove_split_merge.py:   using new similarity functions and other settings for unsupervised SRL
14. results.csv:        record experiments' results

## 4. data format

1. data format for conll09
    0. word id (start from 1)
    1. word (origin word)
    2. word lemma (ground truth lemma for word)
    3. word lemma (predict lemma for word)
    4. POS tag (ground truth pos)
    5. POS tag (predicted pos)
    6. unrelated to SRL
    7. unrelated to SRL
    8. father node in syntax tree (ground truth for dependency header's word_id)
    9. father node in syntax tree (predicted dependency header's word_id)
    10. relation to father node (ground truth dependency relation to its header)
    11. relation to father node (predicted dependency relation to its header)
    12. predicate indicator (Y means it is a predicate, _ means it's not)
    13. is predicate: (if: 12 == Y, then: (predicate lemma).(predicate class) else: _)
    14. argument label of the 1-st predicate
    15. argument label of the 2-nd predicate
    16. argument label of the 3-rd predicate
    ...
2. flattened format
    0. sentence id (start from 0)
    1. predicate id (start from 0)
    2. sentence length
    3. is predicate: 0 or 1
    4. word id (start from 1)
    5. word
    6. lemma(true)
    7. lemma(predict)
    8. POS(true)
    9. POS(predict)
    10. dependency header node id(true)
    11. dependency header node id(predict)
    12. dependency relation(true)
    13. dependency relation(predict)
    14. semantic role