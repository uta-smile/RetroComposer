# this file is adapted from https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/finetune.py

import argparse
import copy
import json
import numpy as np
import os
import torch
import torch.multiprocessing as mp
import torch.optim as optim

from collections import Counter
from tqdm import tqdm
from torch_geometric.data import DataLoader

from gnn import GNN_graphpred
from prepare_mol_graph import ReactionDataset
from chemutils import cano_smiles


def train(args, model, device, loader, optimizer=None, train=True):
    if train:
        model.train()
    else:
        model.eval()

    topk = [5, 10, 20, 30, 40, 50]
    loss_list, res_list = [], [[] for _ in topk]
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        product, reactants = batch
        product = product.to(device)
        reactants = reactants.to(device)
        if model.training:
            loss, logits, probs = model.ranking(product, reactants, args.typed, args.loss)
        else:
            with torch.no_grad():
                loss, logits, probs = model.ranking(product, reactants, args.typed, args.loss)

        loss = torch.stack(loss).mean()
        loss_list.append(loss.item())
        for p in probs:
            for i, k in enumerate(topk):
                res_list[i].append((torch.argmax(p[0, :k]) == 0).item())

        if model.training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    loss = np.mean(loss_list)
    res_list = [np.mean(r) for r in res_list]
    return loss, res_list


def eval_decoding(args, model, device, dataset):
    model.eval()
    pred_results = {}
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        product, reactants = batch
        product = product.to(device)
        reactants = reactants.to(device)
        with torch.no_grad():
            _, logits, probs = model.ranking(product, reactants, args.typed, args.loss)
        for idx, logit in enumerate(logits):
            assert logit.shape[1] == reactants.num_reacts[idx]
            pred_results[product.index[idx]] = (batch[0][idx], batch[1][idx], logit)

    weight = 0.4
    top1 = []
    res_list = []
    for idx, val in pred_results.items():
        product, reactants, logit = val
        log_prob = torch.log_softmax(logit, dim=1).squeeze(dim=0).cpu()
        log_prob_retro = reactants.log_prob.cpu()
        log_prob = weight * log_prob_retro + (1. - weight) * log_prob
        indexes = torch.sort(log_prob[1:], descending=True)[1].tolist()
        found = False
        for k, react_idx in enumerate(indexes):
            if cano_smiles(reactants.smiles[react_idx+1]) == product.reactant_gt:
                res_list.append(k+1)
                found = True
                break
        if not found:
            res_list.append(1000)

    print('\nweight:', weight)
    print('ranking for {} examples'.format(len(res_list)))
    print('pred_results len:', len(pred_results))

    sum = 0
    counter = Counter(res_list)
    top1.append(counter[1] / len(res_list))
    with open(os.path.join(args.filename, 'ranking_results.txt'), 'a') as f:
        f.write('\ninput model checkpoint file: {}\n'.format(args.input_model_file))
        f.write('weight: {}\n'.format(weight))
        f.write('ranking for {} examples, pred_results len = {}\n'.format(len(res_list), len(pred_results)))
        for topk in range(1, 11):
            sum += counter[topk]
            log_line = 'top-{}: {:.6f}'.format(topk, sum / len(res_list))
            f.write(log_line + '\n')
            print(log_line)


def train_multiprocess(rank, args, model, device, train_dataset, valid_dataset):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    epochs = args.epochs // args.num_processes
    output_model_file = os.path.join(args.filename, 'model.pt')
    print('output_model_file:', output_model_file)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    valid_dataset = copy.deepcopy(valid_dataset)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    for epoch in range(1, epochs + 1):
        print("====rank and epoch: ", rank, epoch)
        loss, train_res = train(args, model, device, train_loader, optimizer)
        print("rank: %d epoch: %d train_loss: %.4f top-1 acc by ranking top-5: %.4f top-10: %.4f top-20: %.4f top-30: %.4f top-40: %.4f top-50: %.4f" % (rank, epoch, loss, *train_res))
        scheduler.step()
        print("====evaluation")
        loss, val_res = train(args, model, device, val_loader, train=False)
        print("rank: %d epoch: %d val_loss: %.4f top-1 acc by ranking top-5: %.4f top-10: %.4f top-20: %.4f top-30: %.4f top-40: %.4f top-50: %.4f" % (rank, epoch, loss, *val_res))
        if rank == 0:
            torch.save(model.state_dict(), output_model_file)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='input batch size for training (default: 4)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.00005,
                        help='learning rate (default: 0.00005)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=3,
                        help='number of GNN message passing layers (default: 3).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.2,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--fusion', type=int, default=2,
                        help="node feature fusion function type, 1: -, 2: concat(-, +), 3: concat(-, *)")
    parser.add_argument('--graph_pooling', type=str, default="attention",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="concat",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default='data/USPTO50K', help='root directory of dataset.')
    parser.add_argument('--loss', type=str, default='ce', help='ce or bce')
    parser.add_argument('--atom_feat_dim', type=int, default=45, help="atom feature dimension.")
    parser.add_argument('--bond_feat_dim', type=int, default=12, help="bond feature dimension.")
    parser.add_argument('--typed', action='store_true', default=False, help='if given reaction types')
    parser.add_argument('--test_only', action='store_true', default=False, help='only evaluate on test data')
    parser.add_argument('--multiprocess', action='store_true', default=False, help='train a model with multi process')
    parser.add_argument('--num_processes', type=int, default=2, help='number of processes for multi-process training')
    parser.add_argument('--input_model_file', type=str, default='uspto50k', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default='uspto50k', help='output filename')
    parser.add_argument('--tag', type=str, default='', help='tag name')
    parser.add_argument('--runseed', type=int, default=0, help="Seed for minibatch selection, random initialization.")
    parser.add_argument('--eval_train', action='store_true', default=False, help='evaluating training or not')
    parser.add_argument('--topk', type=int, default=50, help='use topk reactants for training the reranking model.')
    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    # set up dataset
    if args.test_only:
        test_dataset = ReactionDataset(args.dataset, split='test', typed=args.typed, topk=50)
    else:
        train_dataset = ReactionDataset(args.dataset, split='train', typed=args.typed, topk=args.topk)
        valid_dataset = ReactionDataset(args.dataset, split='valid', typed=args.typed, topk=50)

    if args.typed:
        args.atom_feat_dim += 10
        args.filename += '_typed'

    # set up model
    model = GNN_graphpred(args.num_layer, args.emb_dim, args.atom_feat_dim, args.bond_feat_dim,
                          '', args.fusion, 0, 0, JK=args.JK, drop_ratio=args.dropout_ratio,
                          graph_pooling=args.graph_pooling)
    del model.rnn_model
    model.to(device)

    dataset = os.path.basename(args.dataset)
    args.filename = os.path.join('logs', dataset, args.filename, 'ranking/top{}_fusion{}'.format(args.topk, args.fusion))
    if args.tag:
        args.filename += '_' + args.tag
    os.makedirs(args.filename, exist_ok=True)
    print('log filename:', args.filename)
    if not args.input_model_file == "":
        input_model_file = os.path.join(args.filename, args.input_model_file)
        model.from_pretrained(input_model_file, args.device)
        print("load model from:", input_model_file)

    if args.test_only:
        print("evaluate on test data only")
        eval_decoding(args, model, device, test_dataset)
        exit(1)

    if args.multiprocess:
        mp.set_start_method('spawn', force=True)
        model.share_memory()  # gradients are allocated lazily, so they are not shared here
        processes = []
        for rank in range(args.num_processes):
            p = mp.Process(
                target=train_multiprocess,
                args=(rank, args, model, device, train_dataset, valid_dataset)
            )
            # We first train the model across `num_processes` processes
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    else:
        output_model_file = os.path.join(args.filename, 'model.pt')
        print('output_model_file:', output_model_file)
        # set up optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
        print(optimizer)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
        for epoch in range(1, args.epochs + 1):
            print("====epoch " + str(epoch))
            loss, res = train(args, model, device, train_loader, optimizer)
            print(loss)
            print('top-1 acc by ranking top 5, 10, 20, 30, 40, 50:', res)
            scheduler.step()
            torch.save(model.state_dict(), output_model_file)

            print("")
            val_res = train(args, model, device, val_loader, train=False)
            print(val_res)



if __name__ == "__main__":
    main()
