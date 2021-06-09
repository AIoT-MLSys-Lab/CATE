import os
import sys
sys.path.insert(0, os.getcwd())
from pybnn.dngo import DNGO
import argparse
import json
import torch
import numpy as np
import scipy.stats as stats
from collections import defaultdict
from darts.cnn.train_search import Train

def load(path):
    data = torch.load(path)
    print('load pretrained embeddings from {}'.format(path))
    features = data['embeddings']
    genotypes = data['genotypes']
    print('loading finished. pretrained embeddings shape {}'.format(features.shape))
    return features, genotypes

def query(trainer, seed, genotype, epochs):
    rewards, rewards_test = trainer.main(seed, genotype, epochs=epochs, train_portion=args.train_portion, save=args.dataset)
    val_sum = 0
    for epoch, val_acc in rewards:
        val_sum += val_acc
    val_avg = val_sum / len(rewards)
    return val_avg / 100., rewards_test[-1][-1] / 100.

def get_samples(features, genotype, visited, trainer):
    np.random.seed(args.seed)
    init_inds = np.random.permutation(list(range(features.shape[0])))[:args.init_size]
    init_inds = torch.Tensor(init_inds).long()
    init_feat_samples = features[init_inds]
    init_geno_samples = [genotype[i.item()] for i in init_inds]
    init_valid_label_samples = []
    init_test_label_samples = []

    for geno in init_geno_samples:
        val_acc, test_acc = query(trainer, args.seed, geno, args.inner_epochs)
        init_valid_label_samples.append(val_acc)
        init_test_label_samples.append(test_acc)

    init_valid_label_samples = torch.Tensor(init_valid_label_samples)
    init_test_label_samples = torch.Tensor(init_test_label_samples)
    for idx in init_inds:
        visited[idx.item()] = True
    return init_feat_samples, init_geno_samples, init_valid_label_samples, init_test_label_samples, visited

def step(q, features, genotype, visited, trainer):
    dist = torch.norm(features - q.view(1, -1), dim=1)
    knn = (-1 * dist).topk(dist.shape[0])
    min_dist, min_idx = knn.values, knn.indices
    i = 0
    while True:
        if len(visited) == dist.shape[0]:
            print("cannot find in the dataset")
            exit()
        if min_idx[i].item() not in visited:
            visited[min_idx[i].item()] = True
            break
        i += 1
    geno = genotype[min_idx[i].item()]
    val_acc, test_acc = query(trainer, args.seed, geno, args.inner_epochs)

    return features[min_idx[i].item()], geno, torch.tensor(val_acc), torch.tensor(test_acc), visited

def computation_aware_search(label_next_valid, feat_samples, valid_label_samples,
                             test_label_samples, visited, best_trace, counter,
                             topk, features, genotype, CURR_BEST_VALID,
                             CURR_BEST_TEST, CURR_BEST_GENOTYPE, MAX_BUDGET, trainer):

    indices = torch.argsort(valid_label_samples.view(-1))
    for ind in indices[-topk:]:
        if valid_label_samples[ind] not in label_next_valid:
            feat_nn, geno_nn, valid_label_nn, test_label_nn, visited = \
                step(feat_samples[ind], features, genotype, visited, trainer)
            if valid_label_nn.item() > CURR_BEST_VALID:
                print('FIND BEST VALID FROM NNS')
                CURR_BEST_VALID = valid_label_nn.item()
                CURR_BEST_TEST = test_label_nn.item()
                CURR_BEST_GENOTYPE = geno_nn
            feat_samples = torch.cat((feat_samples, feat_nn.view(1, -1)), dim=0)
            valid_label_samples = torch.cat((valid_label_samples.view(-1, 1), valid_label_nn.view(1, 1)), dim=0)
            test_label_samples = torch.cat((test_label_samples.view(-1, 1), test_label_nn.view(1, 1)), dim=0)
            counter += 1
            best_trace['validation_acc'].append(float(CURR_BEST_VALID))
            best_trace['test_acc'].append(float(CURR_BEST_TEST))
            best_trace['genotype'].append(str(CURR_BEST_GENOTYPE))
            best_trace['counter'].append(counter)
            if counter > MAX_BUDGET:
                break
    return feat_samples, valid_label_samples, test_label_samples, \
           visited, best_trace, counter, CURR_BEST_VALID, \
           CURR_BEST_TEST, CURR_BEST_GENOTYPE

def propose_location(ei, features, genotype, visited, trainer):
    k = args.topk
    c = 0
    print('remaining length of indices set:', len(features) - len(visited))
    indices = torch.argsort(ei)
    ind_dedup = []
    # remove visited indices
    for idx in reversed(indices):
        if c == k:
            break
        if idx.item() not in visited:
            visited[idx.item()] = True
            ind_dedup.append(idx.item())
            c += 1
    ind_dedup = torch.Tensor(ind_dedup).long()
    proposed_x = features[ind_dedup]
    proposed_geno = [genotype[i.item()] for i in ind_dedup]
    proposed_val_acc = []
    proposed_test_acc = []
    for geno in proposed_geno:
        val_acc, test_acc = query(trainer, args.seed, geno, args.inner_epochs)
        proposed_val_acc.append(val_acc)
        proposed_test_acc.append(test_acc)

    return proposed_x, proposed_geno, torch.Tensor(proposed_val_acc), torch.Tensor(proposed_test_acc), visited

def expected_improvement_search(features, genotype):
    """ implementation of CATE-DNGO-LS on the DARTS search space """
    CURR_BEST_VALID = 0.
    CURR_BEST_TEST = 0.
    CURR_BEST_GENOTYPE = None
    PREV_BEST = 0
    MAX_BUDGET = args.max_budgets
    window_size = 1024
    round = 0
    counter = 0
    visited = {}
    best_trace = defaultdict(list)
    trainer = Train()
    feat_samples, geno_samples, valid_label_samples, test_label_samples, visited = get_samples(features, genotype, visited, trainer)

    for feat, geno, acc_valid, acc_test in zip(feat_samples, geno_samples, valid_label_samples, test_label_samples):
        counter += 1
        if acc_valid > CURR_BEST_VALID:
            CURR_BEST_VALID = acc_valid
            CURR_BEST_TEST = acc_test
            CURR_BEST_GENOTYPE = geno
        best_trace['validation_acc'].append(float(CURR_BEST_VALID))
        best_trace['test_acc'].append(float(CURR_BEST_TEST))
        best_trace['genotype'].append(str(CURR_BEST_GENOTYPE))
        best_trace['counter'].append(counter)

    while counter < MAX_BUDGET:
        if round == args.rounds:
            feat_samples, geno_samples, valid_label_samples, test_label_samples, visited = get_samples(features, genotype, visited, trainer)
            for feat, geno, acc_valid, acc_test in zip(feat_samples, geno_samples, valid_label_samples, test_label_samples):
                counter += 1
                if acc_valid > CURR_BEST_VALID:
                    CURR_BEST_VALID = acc_valid
                    CURR_BEST_TEST = acc_test
                    CURR_BEST_GENOTYPE = geno
                best_trace['validation_acc'].append(float(CURR_BEST_VALID))
                best_trace['test_acc'].append(float(CURR_BEST_TEST))
                best_trace['genotype'].append(str(CURR_BEST_GENOTYPE))
                best_trace['counter'].append(counter)
            round = 0
        print("current counter: {}, best validation acc.: {}, test acc.: {}".format(counter, CURR_BEST_VALID, CURR_BEST_TEST))
        print("current best genotype: {}".format(CURR_BEST_GENOTYPE))
        model = DNGO(num_epochs=30, n_units=128, do_mcmc=False, normalize_output=False)
        model.train(X=feat_samples.numpy(), y=valid_label_samples.view(-1).numpy(), do_optimize=True)
        print(model.network)
        m = []
        v = []
        chunks = int(features.shape[0] / window_size)
        if features.shape[0] % window_size > 0:
            chunks += 1
        features_split = torch.split(features, window_size, dim=0)
        for i in range(chunks):
            m_split, v_split = model.predict(features_split[i].numpy())
            m.extend(list(m_split))
            v.extend(list(v_split))
        mean = torch.Tensor(m)
        sigma = torch.Tensor(v)
        u = (mean - torch.Tensor([args.objective]).expand_as(mean)) / sigma
        ei = sigma * (u * stats.norm.cdf(u) + 1 + stats.norm.pdf(u))
        feat_next, geno_next, label_next_valid, label_next_test, visited = \
            propose_location(ei, features, genotype, visited, trainer)

        # add proposed networks to the pool
        for feat, geno, acc_valid, acc_test in zip(feat_next, geno_next, label_next_valid, label_next_test):
            if acc_valid.item() > CURR_BEST_VALID:
                print('FIND BEST VALID FROM DNGO')
                CURR_BEST_VALID = acc_valid.item()
                CURR_BEST_TEST = acc_test.item()
                CURR_BEST_GENOTYPE = geno
            feat_samples = torch.cat((feat_samples, feat.view(1, -1)), dim=0)
            geno_samples.append(geno)
            valid_label_samples = torch.cat((valid_label_samples.view(-1, 1), acc_valid.view(1, 1)), dim=0)
            test_label_samples = torch.cat((test_label_samples.view(-1, 1), acc_test.view(1, 1)), dim=0)
            counter += 1
            best_trace['validation_acc'].append(float(CURR_BEST_VALID))
            best_trace['test_acc'].append(float(CURR_BEST_TEST))
            best_trace['genotype'].append(str(CURR_BEST_GENOTYPE))
            best_trace['counter'].append(counter)

            if counter > MAX_BUDGET:
                break

        if args.computation_aware_search:
            feat_samples, valid_label_samples, test_label_samples, visited, best_trace, counter, CURR_BEST_VALID, CURR_BEST_TEST, CURR_BEST_GENOTYPE = \
            computation_aware_search(label_next_valid, feat_samples,
                                     valid_label_samples, test_label_samples,
                                     visited, best_trace, counter, args.topk,
                                     features, genotype, CURR_BEST_VALID,
                                     CURR_BEST_TEST, CURR_BEST_GENOTYPE,
                                     MAX_BUDGET, trainer)
        if PREV_BEST < CURR_BEST_VALID:
            PREV_BEST = CURR_BEST_VALID
        else:
            round += 1

    res = dict()
    res['validation_acc'] = best_trace['validation_acc']
    res['test_acc'] = best_trace['test_acc']
    res['genotype'] = best_trace['genotype']
    res['counter'] = best_trace['counter']
    save_path = args.dataset + '/' + args.output_path + '/' + 'dim{}'.format(args.dim)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    print('save to {}'.format(save_path))
    fh = open(os.path.join(save_path, 'run_{}.json'.format(args.seed)), 'w')
    json.dump(res, fh)
    fh.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CATE-DNGO-LS")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument('--dim', type=int, default=64, help='feature dimension')
    parser.add_argument('--objective', type=float, default=1.0, help='ei objective')
    parser.add_argument('--init_size', type=int, default=16, help='init samples')
    parser.add_argument('--topk', type=int, default=5, help='acquisition samples')
    parser.add_argument('--rounds', type=int, default=10, help='rounds allowed for local minimum')
    parser.add_argument('--inner_epochs', type=int, default=50, help='inner loop epochs')
    parser.add_argument('--train_portion', type=float, default=0.9, help='inner loop train/val split')
    parser.add_argument('--max_budgets', type=int, default=300, help='max number of trials')
    parser.add_argument('--dataset', type=str, default='darts', help='darts')
    parser.add_argument('--output_path', type=str, default='bo/', help='bo')
    parser.add_argument('--embedding_path', type=str, default='cate_darts.pt')
    parser.add_argument('--computation_aware_search', type=bool, default=True)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    features, genotype = load(args.embedding_path)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    expected_improvement_search(features, genotype)