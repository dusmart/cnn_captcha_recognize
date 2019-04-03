import os

raw_data_path = os.path.join(os.path.dirname(__file__), "raw_data")
flattened_data_path = os.path.join(os.path.dirname(__file__), "flattened_data")
labels_path = os.path.join(os.path.dirname(__file__), "labels")


raw_train_data_path = os.path.join(raw_data_path, "conll09_train.dataset")
raw_test_data_path = os.path.join(raw_data_path, "conll09_test.dataset")
raw_dev_data_path = os.path.join(raw_data_path, "conll09_dev.dataset")

flattened_train_data_path = os.path.join(flattened_data_path, "conll09_train.dataset")
flattened_test_data_path = os.path.join(flattened_data_path, "conll09_test.dataset")
flattened_dev_data_path = os.path.join(flattened_data_path, "conll09_dev.dataset")

pos_vocab_path = os.path.join(labels_path, "pos_vocab")
pos2id_path = os.path.join(labels_path, "pos2id")
id2pos_path = os.path.join(labels_path, "id2pos")

deprel_vocab_path = os.path.join(labels_path, "deprel_vocab")
deprel2id_path = os.path.join(labels_path, "deprel2id")
id2deprel_path = os.path.join(labels_path, "id2deprel")

arg_vocab_path = os.path.join(labels_path, "arg_vocab")
arg2id_path = os.path.join(labels_path, "arg2id")
id2arg_path = os.path.join(labels_path, "id2arg")

preposition_vocab_path = os.path.join(labels_path, "preposition_vocab")
preposition2id_path = os.path.join(labels_path, "preposition2id")
id2preposition_path = os.path.join(labels_path, "id2preposition")



