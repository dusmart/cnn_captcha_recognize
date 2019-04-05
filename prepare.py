from parameters import *
from data_utils import *

flat_dataset(raw_train_data_path, flattened_train_data_path)
flat_dataset(raw_test_data_path, flattened_test_data_path)
flat_dataset(raw_dev_data_path, flattened_dev_data_path)
flat_dataset(raw_sample_data_path, flattened_sample_data_path)

print('\n-- shrink pretrained embeding --')
shrink_pretrained_embedding(raw_train_data_path, raw_dev_data_path, raw_test_data_path,
                            pretrained_glove_100, pretrained_emb_size, pretrained_embed_path,
                            id2pretrained_path, pretrained2id_path, pretrained_vocab_path)


vocab_maker = VocabMaker()
vocab_maker.make_vocab(raw_train_data_path, deprel_vocab_path, deprel2id_path, id2deprel_path, "deprel")
vocab_maker.make_vocab(raw_train_data_path, arg_vocab_path, arg2id_path, id2arg_path, "arg")
vocab_maker.make_vocab(raw_train_data_path, pos_vocab_path, pos2id_path, id2pos_path, "pos")
vocab_maker.make_vocab(raw_train_data_path, preposition_vocab_path, preposition2id_path, id2preposition_path, "preposition", freq_lower_bound=500, use_lower_bound=True)
vocab_maker.make_vocab(raw_train_data_path, arghead_vocab_path, arghead2id_path, id2arghead_path, "arghead")
