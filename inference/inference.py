import os
import sys
sys.path.insert(0, os.getcwd())
import torch
import numpy as np
import torch.nn.functional as F
from parameterLoader import argLoader
from layers import PairWiseLearning, GraphEncoder
from utils.utils import floyed

def prepare_graph(graph, config):
    if config.dataset.split('_')[0] == 'nasbench101':
        Xs, Rs, valid_accs, test_accs, times = zip(*graph)
    elif config.dataset.split('_')[0] == 'nasbench301':
        Xs, Rs, genos, predicted_accs, predicted_runtimes = zip(*graph)
    elif config.dataset.split('_')[0] == 'darts':
        Xs, Rs, genos = zip(*graph)
    elif config.dataset.split('_')[0] == 'oo':
        Xs, Rs, valid_accs, test_accs, times = zip(*graph)
    else:
        raise NotImplementedError()

    ls = [len(it) for it in Xs]
    maxL = max(ls)
    inputs = []
    masks = []

    for x, r, l in zip(Xs, Rs, ls):
        input_i = torch.LongTensor(x)
        mask_i = torch.from_numpy(floyed(r)).float()
        #mask_i = torch.from_numpy(np.asarray(r)).float() # no floyed
        padded_input_i = F.pad(input_i, (0, maxL - l), "constant", config.PAD)
        padded_mask_i = F.pad(mask_i, (0, maxL - mask_i.shape[1], 0, maxL - mask_i.shape[1]), "constant", config.PAD)
        inputs.append(padded_input_i)
        masks.append(padded_mask_i)

    if config.dataset.split('_')[0] == 'nasbench101':
        return torch.stack(inputs), torch.stack(masks), valid_accs, test_accs, times
    elif config.dataset.split('_')[0] == 'nasbench301':
        return torch.stack(inputs), torch.stack(masks), genos, predicted_accs, predicted_runtimes
    elif config.dataset.split('_')[0] == 'darts':
        return torch.stack(inputs), torch.stack(masks), genos
    elif config.dataset.split('_')[0] == 'oo':
        return torch.stack(inputs), torch.stack(masks), valid_accs, test_accs, times
    else:
        raise NotImplementedError()

def inference(config):
    # Model
    net = PairWiseLearning(config)
    if torch.cuda.is_available():
        net = net.cuda(config.device)

    # Convert DataParallel when testing
    pretrained_dict = torch.load(config.pretrained_path, map_location="cuda:{}".format(config.device))['state_dict']
    pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
    net.load_state_dict(pretrained_dict)

    # Inference Parameters
    data = []
    trainSet = torch.load(config.train_data)
    validSet = torch.load(config.valid_data)
    for dataset in [trainSet, validSet]:
        for i in range(len(dataset)):
            R = dataset[i]['adj']
            X = np.argmax(np.asarray(dataset[i]['ops']), axis=-1)
            if config.dataset.split('_')[0] == 'nasbench101':
                valid_acc = dataset[i]['validation_accuracy']
                test_acc = dataset[i]['test_accuracy']
                time = dataset[i]['training_time']
                data.append([X, R, valid_acc, test_acc, time])
            elif config.dataset.split('_')[0] == 'nasbench301':
                genotype = dataset[i]['genotype']
                predicted_acc = dataset[i]['predicted_acc']
                predicted_runtime = dataset[i]['predicted_runtime']
                data.append([X, R, genotype, predicted_acc, predicted_runtime])
            elif config.dataset.split('_')[0] == 'darts':
                genotype = dataset[i]['genotypes']
                data.append([X, R, genotype])
            elif config.dataset.split('_')[0] == 'oo':
                valid_acc = dataset[i]['validation_acc']
                test_acc = dataset[i]['test_acc']
                time = dataset[i]['training_time']
                data.append([X, R, valid_acc, test_acc, time])
            else:
                raise NotImplementedError()

    if config.dataset.split('_')[0] == 'nasbench101':
        X, maskX, valid_accs, test_accs, times = prepare_graph(data, config)
    elif config.dataset.split('_')[0] == 'nasbench301':
        X, maskX, genotypes, predicted_accs, predicted_runtimes = prepare_graph(data, config)
    elif config.dataset.split('_')[0] == 'darts':
        X, maskX, genotypes = prepare_graph(data, config)
    elif config.dataset.split('_')[0] == 'oo':
        X, maskX, valid_accs, test_accs, times = prepare_graph(data, config)
    else:
        raise NotImplementedError()

    maskX_ = maskX.transpose(-2, -1)

    # Inference
    net.eval()
    dropout = torch.nn.Dropout(p=config.dropout)
    dropout.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(data), config.batch_size):
            print('data {} / {}'.format(i, len(data)))
            bs = min(args.batch_size, len(data) - i)
            x = X[i: i+bs].cuda(config.device)
            m = maskX[i: i+bs].cuda(config.device)
            m_ = maskX_[i: i+bs].cuda(config.device)
            emb_x = dropout(net.opEmb(x))
            h_x = net.graph_encoder(emb_x, m, m_)
            embeddings.append(GraphEncoder.get_embeddings(h_x))

    embeddings = torch.cat(embeddings, dim=0)
    if config.dataset.split('_')[0] == 'nasbench101':
        pretrained_embeddings = {'embeddings': embeddings, 'valid_accs': valid_accs, 'test_accs': test_accs, 'times': times}
    elif config.dataset.split('_')[0] == 'nasbench301':
        pretrained_embeddings = {'embeddings': embeddings, 'genotypes': genotypes, 'predicted_accs': predicted_accs, 'predicted_runtimes': predicted_runtimes}
    elif config.dataset.split('_')[0] == 'darts':
        pretrained_embeddings = {'embeddings': embeddings, 'genotypes': genotypes}
    elif config.dataset.split('_')[0] == 'oo':
        pretrained_embeddings = {'embeddings': embeddings, 'valid_accs': valid_accs, 'test_accs': test_accs, 'times': times}
    else:
        raise NotImplementedError()

    torch.save(pretrained_embeddings, 'cate_' + config.dataset + '.pt')

if __name__ == '__main__':
    args = argLoader()
    torch.cuda.set_device(args.device)
    inference(args)