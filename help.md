1. data format for conll09
    0. word id
    1. word
    2. word lemma (ground truth)
    3. word lemma (predict)
    4. POS tag (ground truth)
    5. POS tag (predict)
    6. unrelated to SRL
    7. unrelated to SRL
    8. father node in syntax tree (ground truth)
    9. father node in syntax tree (predict)
    10. relation to father node (ground truth)
    11. relation to father node (predict)
    12. predicate indicator (Y means it is a predicate)
    13. if: 12 == Y, then: (predicate lemma).(predicate class) else: _
    14. argument label of the 1-st predicate
    15. argument label of the 2-nd predicate
    16. argument label of the 3-rd predicate
    ...
2. flattened format
    0. sentence id
    1. predicate id
    2. sentence length
    3. is predicate: 0 or 1
    4. word id
    5. word
    6. lemma(true)
    7. lemma(predict)
    8. POS(true)
    9. POS(predict)
    10. father node id(true)
    11. father node id(predict)
    12. dependency relation(true)
    13. dependency relation(predict)
    14. semantic role
