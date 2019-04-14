from glove_split_merge import *


def got_best_alpha():
    Cluster.GAMMA = 0.95
    best_pre, best_coll, best_f1, best_alpha = 0, 0, 0, 0
    truths, predicts = split_phase(flattened_train_data_path)
    split_pre, split_coll, split_f1 = evaluation(truths, predicts)
    print("====split====", split_pre, split_coll, split_f1)
    for alpha in np.arange(0.8, 1, 0.02):
        final_pre = merge_phases(deepcopy(predicts), alpha)
        pre, coll, f1 = evaluation(truths, final_pre)
        print("====merge {}====".format(alpha), pre, coll, f1)
        if f1 > best_f1:
            best_pre, best_coll, best_f1, best_alpha = pre, coll, f1, alpha
        elif f1 < best_f1-0.2:
            break
        if pre < coll and f1 < split_f1:
            print("-----failed-----")
            break

    print("=====best {}======".format(best_alpha), best_pre, best_coll, best_f1)

if __name__ == "__main__":
    got_best_alpha()
