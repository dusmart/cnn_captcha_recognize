import os

# the conll09 data format, see readme for details
# suffix _p stand for predicted, _g stand for ground thruth
ID2CONLL = ['word_id', 'word', 'lemma_g', 'lemma_p', 'pos_g', 'pos_p', 
            'unk_1', 'unk_2', 'dephead_g', 'dephead_p', 
            'deprel_g', 'deprel_p', 'is_predicate']
CONLL2ID = {info: idx for idx, info in enumerate(ID2CONLL)}

# the flattened conll09 data format, see readme for details
# suffix _p stand for predicted, _g stand for ground thruth
ID2FLATTEN = ["sent_id","predicate_id","sent_len","is_predicate",
            "word_id","word","lemma_g","lemma_p","pos_g","pos_p",
            "dephead_g","dephead_p","deprel_g","deprel_p","sr"]
FLATTEN2ID = {info: idx for idx, info in enumerate(ID2FLATTEN)}

raw_data_path = os.path.join(os.path.dirname(__file__), "raw_data")
flattened_data_path = os.path.join(os.path.dirname(__file__), "flattened_data")
labels_path = os.path.join(os.path.dirname(__file__), "labels")
pretrained_model_path = os.path.join(os.path.dirname(__file__), "pretrained_model")


pretrained_glove_100 = os.path.join(pretrained_model_path, "glove.100d.txt")
pretrained_emb_size = 100

raw_train_data_path = os.path.join(raw_data_path, "conll09_train.dataset")
raw_test_data_path = os.path.join(raw_data_path, "conll09_test.dataset")
raw_dev_data_path = os.path.join(raw_data_path, "conll09_dev.dataset")
raw_sample_data_path = os.path.join(raw_data_path, "conll09_sample.dataset")

flattened_train_data_path = os.path.join(flattened_data_path, "conll09_train.dataset")
flattened_test_data_path = os.path.join(flattened_data_path, "conll09_test.dataset")
flattened_dev_data_path = os.path.join(flattened_data_path, "conll09_dev.dataset")
flattened_sample_data_path = os.path.join(flattened_data_path, "conll09_sample.dataset")

pos_vocab_path = os.path.join(labels_path, "pos_vocab")
pos2id_path = os.path.join(labels_path, "pos2id")
id2pos_path = os.path.join(labels_path, "id2pos")

argpos_vocab_path = os.path.join(labels_path, "argpos_vocab")
argpos2id_path = os.path.join(labels_path, "argpos2id")
id2argpos_path = os.path.join(labels_path, "id2argpos")

deprel_vocab_path = os.path.join(labels_path, "deprel_vocab")
deprel2id_path = os.path.join(labels_path, "deprel2id")
id2deprel_path = os.path.join(labels_path, "id2deprel")

arg_vocab_path = os.path.join(labels_path, "arg_vocab")
arg2id_path = os.path.join(labels_path, "arg2id")
id2arg_path = os.path.join(labels_path, "id2arg")

preposition_vocab_path = os.path.join(labels_path, "preposition_vocab")
preposition2id_path = os.path.join(labels_path, "preposition2id")
id2preposition_path = os.path.join(labels_path, "id2preposition")

arghead_vocab_path = os.path.join(labels_path, "arghead_vocab")
arghead2id_path = os.path.join(labels_path, "arghead2id")
id2arghead_path = os.path.join(labels_path, "id2arghead")

argspan_vocab_path = os.path.join(labels_path, "argspan_vocab")
argspan2id_path = os.path.join(labels_path, "argspan2id")
id2argspan_path = os.path.join(labels_path, "id2argspan")

id2pretrained_path = os.path.join(pretrained_model_path, "id2pretrained")
pretrained2id_path = os.path.join(pretrained_model_path, "pretrained2id")
pretrained_embed_path = os.path.join(pretrained_model_path, "pretrained.emb")
pretrained_vocab_path = os.path.join(pretrained_model_path, "pretrained_vocab")