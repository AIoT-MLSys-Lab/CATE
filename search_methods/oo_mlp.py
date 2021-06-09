import os
import sys
sys.path.insert(0, os.getcwd())
import argparse
import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import defaultdict

class MLP(nn.Module):
    def __init__(self, input_dim, n_units):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, n_units, bias=False)
        self.fc2 = nn.Linear(n_units, 1, bias=False)
        self.relu = F.relu

    def forward(self, inputs):
        out = self.fc1(inputs)
        out = torch.nn.functional.dropout(out, 0.2)
        out = self.fc2(out).view(-1)
        return out

def load(path):
    data = torch.load(path)
    print('load pretrained embeddings from {}'.format(path))
    features = data['embeddings']
    valid_labels = data['valid_accs']
    test_labels = data['test_accs']
    training_time = data['times']
    test_labels = torch.Tensor(test_labels)
    valid_labels = torch.Tensor(valid_labels)
    training_time = torch.Tensor(training_time)
    print('loading finished. pretrained embeddings shape {}'.format(features.shape))
    return features, valid_labels, test_labels, training_time


def get_init_samples(features, valid_labels, test_labels, training_time, visited):
    np.random.seed(args.seed)
    init_inds = np.random.permutation(list(range(features.shape[0])))[:args.init_size]
    init_inds = torch.Tensor(init_inds).long()
    init_feat_samples = features[init_inds]
    init_valid_label_samples = valid_labels[init_inds]
    init_test_label_samples = test_labels[init_inds]
    init_time_samples = training_time[init_inds]
    for idx in init_inds:
        visited[idx] = True
    return init_feat_samples, init_valid_label_samples, init_test_label_samples, \
           init_time_samples, visited


def propose_location(predicted_labels, features, valid_labels, test_labels, training_time, visited):
    k = args.topk
    predicted_labels = predicted_labels.view(-1)
    print('remaining length of indices set:', len(features) - len(visited))
    indices = torch.argsort(predicted_labels)[-k:]
    ind_dedup = []
    for idx in indices:
        if idx not in visited:
            visited[idx] = True
            ind_dedup.append(idx)
    ind_dedup = torch.Tensor(ind_dedup).long()
    proposed_x, proposed_y_valid, proposed_y_test, proposed_time = features[ind_dedup], valid_labels[ind_dedup], test_labels[ind_dedup], training_time[ind_dedup]
    return proposed_x, proposed_y_valid, proposed_y_test, proposed_time, visited


def neural_predictor():
    """ implementation of a 2-layer MLP predictor """
    BEST_TEST_ACC = 0.943175752957662 
    BEST_VALID_ACC = 0.9505542318026224
    CURR_BEST_VALID = 0.
    CURR_BEST_TEST = 0.
    MAX_BUDGET = 150
    window_size = 1024
    counter = 0
    rt = 0.
    visited = {}
    best_trace = defaultdict(list)
    features, valid_labels, test_labels, training_time = load(args.embedding_path)
    feat_samples, valid_label_samples, test_label_samples, time_samples, visited = get_init_samples(features, valid_labels, test_labels, training_time, visited)

    for feat, acc_valid, acc_test, t in zip(feat_samples, valid_label_samples, test_label_samples, time_samples):
        counter += 1
        rt += t.item()
        if acc_valid > CURR_BEST_VALID:
            CURR_BEST_VALID = acc_valid
            CURR_BEST_TEST = acc_test
        best_trace['regret_validation'].append(float(BEST_VALID_ACC - CURR_BEST_VALID))
        best_trace['regret_test'].append(float(BEST_TEST_ACC - CURR_BEST_TEST))
        best_trace['time'].append(rt)
        best_trace['counter'].append(counter)

    while counter <= MAX_BUDGET:
        print("feat_samples:", feat_samples.shape)
        print("valid label_samples:", valid_label_samples.shape)
        print("test label samples:", test_label_samples.shape)
        print("current best validation: {}".format(CURR_BEST_VALID))
        print("current best test: {}".format(CURR_BEST_TEST))
        print("counter: {}".format(counter))
        print("rt: {}".format(rt))
        print(feat_samples.shape)
        print(valid_label_samples.shape)
        model = MLP(input_dim=args.dim, n_units=128).cuda()
        batch_size = args.topk
        epochs = 100
        optimizer = optim.AdamW(model.parameters(), lr=1e-2, eps=1e-8)
        nb = int(feat_samples.shape[0] / batch_size)
        if feat_samples.shape[0] % batch_size > 0:
            nb += 1
        feat_samples_split = torch.split(feat_samples, batch_size, dim=0)
        valid_label_samples_split = torch.split(valid_label_samples, batch_size, dim=0)
        # Start training
        for _ in range(epochs):
            running_loss = 0
            for i in range(nb):
                inputs = feat_samples_split[i].cuda()
                targets = valid_label_samples_split[i].cuda()
                optimizer.zero_grad()
                output = model(inputs)
                loss = torch.nn.functional.mse_loss(output, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                #if i % 10 == 0:
                #    print("Epoch {} of {}".format(epoch + 1, epochs))
                #    print("Running loss:\t\t{:.5g}".format(running_loss / 10))
        y_predict_list = []
        chunks = int(features.shape[0] / window_size)
        if features.shape[0] % window_size > 0:
            chunks += 1
        features_split = torch.split(features, window_size, dim=0)
        for i in range(chunks):
            y_predict_list.append(model(features_split[i].cuda()))

        predicted_labels = torch.cat(y_predict_list)
        feat_next, label_next_valid, label_next_test, time_next, visited = \
            propose_location(predicted_labels, features, valid_labels, test_labels, training_time, visited)
        print('proposed topk: {}'.format(label_next_valid))

        # add proposed networks to the pool
        for feat, acc_valid, acc_test, t in zip(feat_next, label_next_valid, label_next_test, time_next):
            if acc_valid > CURR_BEST_VALID:
                print('FIND BEST VALID FROM NP')
                CURR_BEST_VALID = acc_valid
                CURR_BEST_TEST = acc_test
            feat_samples = torch.cat((feat_samples, feat.view(1, -1)), dim=0)
            valid_label_samples = torch.cat((valid_label_samples.view(-1, 1), acc_valid.view(1, 1)), dim=0)
            test_label_samples = torch.cat((test_label_samples.view(-1, 1), acc_test.view(1, 1)), dim=0)
            counter += 1
            rt += t.item()
            best_trace['regret_validation'].append(float(BEST_VALID_ACC - CURR_BEST_VALID))
            best_trace['regret_test'].append(float(BEST_TEST_ACC - CURR_BEST_TEST))
            best_trace['time'].append(rt)
            best_trace['counter'].append(counter)
            if counter >= MAX_BUDGET:
                break


    res = dict()
    res['regret_validation'] = best_trace['regret_validation']
    res['regret_test'] = best_trace['regret_test']
    res['runtime'] = best_trace['time']
    res['counter'] = best_trace['counter']
    save_path = args.dataset + '/' + args.output_path + '/' + 'dim{}'.format(args.dim)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    print('save to {}'.format(save_path))
    fh = open(os.path.join(save_path, 'run_{}.json'.format(args.seed)),'w')
    json.dump(res, fh)
    fh.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="neural predictor")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument('--dim', type=int, default=64, help='feature dimension')
    parser.add_argument('--init_size', type=int, default=16, help='init samples')
    parser.add_argument('--topk', type=int, default=5, help='topk predicted samples')
    parser.add_argument('--output_path', type=str, default='np', help='np')
    parser.add_argument('--embedding_path', type=str, default='cate_oo_nasbench101.pt | adj_oo_nasbench101.pt')
    parser.add_argument('--dataset', type=str, default='nasbench')
    args = parser.parse_args()
    torch.set_num_threads(1)
    neural_predictor()