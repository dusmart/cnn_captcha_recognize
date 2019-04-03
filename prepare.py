from parameters import *
from data_utils import *

#flat_dataset(raw_train_data_path, flattened_train_data_path)
#flat_dataset(raw_test_data_path, flattened_test_data_path)
#flat_dataset(raw_dev_data_path, flattened_dev_data_path)


#make_deprel_vocab(raw_train_data_path, deprel_vocab_path, deprel2id_path, id2deprel_path)

#make_arg_vocab(raw_train_data_path, raw_dev_data_path, raw_test_data_path, arg_vocab_path, arg2id_path, id2arg_path)

make_preposition_vocab(raw_train_data_path, preposition_vocab_path, preposition2id_path, id2preposition_path)