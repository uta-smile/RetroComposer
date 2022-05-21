# this file is adapted from https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/finetune.py

import argparse
import json
import numpy as np
import copy
import os
import torch
import torch.multiprocessing as mp
import torch.optim as optim

from datetime import datetime
from tqdm import tqdm
from rdkit import Chem
from torch_geometric.data import DataLoader

from gnn import GNN_graphpred
from prepare_mol_graph import MoleculeDataset
from chemutils import cano_smiles


def train(args, model, device, loader, optimizer=None, train=True):
    if train:
        model.train()
    else:
        model.eval()

    loss_list, loss_prod_list, loss_react_list = [], [], []
    prod_pred_res_max, react_pred_res, react_pred_res_each = [], [], []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        loss_prod, loss_react, prod_pred_max, react_pred = model(batch, typed=args.typed)
        loss = loss_prod + loss_react
        loss_list.append(loss.item())
        loss_prod_list.append(loss_prod.item())
        loss_react_list.append(loss_react.item())
        if model.training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        prod_pred_res_max.extend(prod_pred_max)
        react_pred_res_each.extend(react_pred.reshape(-1, ).tolist())
        for react in react_pred:
            react_pred_res.append(False not in react)

    loss = np.mean(loss_list)
    loss_prod = np.mean(loss_prod_list)
    loss_react = np.mean(loss_react_list)
    prod_pred_acc_max = np.mean(prod_pred_res_max)
    react_pred_acc_each = np.mean(react_pred_res_each)
    react_pred_acc = np.mean(react_pred_res)
    return loss, loss_prod, loss_react, prod_pred_acc_max, react_pred_acc_each, react_pred_acc


def eval_decoding(args, model, device, dataset, save_res=True, k=0):
    model.eval()
    pred_results = {}
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    # import pdb; pdb.set_trace()
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        with torch.no_grad():
            beam_nodes = model(batch, typed=args.typed, decode=True, beam_size=args.beam_size)
            cano_pred_mols = {}
            for node in beam_nodes:
                batch_idx = node.index
                data_idx = batch.index[batch_idx]
                if data_idx not in cano_pred_mols:
                    cano_pred_mols[data_idx] = set()
                if data_idx not in pred_results:
                    pred_results[data_idx] = {
                        'rank': 1000,
                        'product': batch.product[batch_idx],
                        'reactant': batch.reactant[batch_idx],
                        'cano_reactants': batch.cano_reactants[batch_idx],
                        'type': batch.type[batch_idx].item(),
                        'seq_gt': batch.sequences[batch_idx],
                        'templates': batch.templates[batch_idx],
                        'templates_pred': [],
                        'templates_pred_log_prob': [],
                        'reactants_pred': [],
                        'seq_pred': [],
                    }
                # import ipdb; ipdb.set_trace()
                product = pred_results[data_idx]['product']
                seq_pred = node.targets_predict
                prod_smarts_list = []
                for i, cand in enumerate(batch.reaction_center_cands[batch_idx]):
                    if cand == seq_pred[0]:
                        prod_smarts_list.extend(batch.reaction_center_cands_smarts[batch_idx][i])
                prod_smarts_list = set(prod_smarts_list)
                assert len(prod_smarts_list)
                # keep product index unchanged, remove padding reactant indexes
                seq_pred[1:] = [tp for tp in seq_pred[1:] if tp < len(dataset.react_smarts_list)]
                decoded_results = dataset.decode_reactant_from_seq(product, seq_pred, prod_smarts_list, keep_mapnums=True)
                for decoded_result in decoded_results:
                    pred_tmpl, pred_mols = decoded_result
                    for pred_mol in pred_mols:
                        cano_pred_mol = cano_smiles(pred_mol)
                        if cano_pred_mol not in cano_pred_mols[data_idx]:
                            cano_pred_mols[data_idx].add(cano_pred_mol)
                            pred_results[data_idx]['templates_pred_log_prob'].append(node.log_prob.item())
                            pred_results[data_idx]['templates_pred'].append(pred_tmpl)
                            pred_results[data_idx]['reactants_pred'].append(pred_mol)
                            pred_results[data_idx]['seq_pred'].append(seq_pred)
                            if pred_results[data_idx]['cano_reactants'] == cano_pred_mol:
                                pred_results[data_idx]['rank'] = min(pred_results[data_idx]['rank'], len(pred_results[data_idx]['seq_pred']))

            beam_nodes.clear()

    print('rank: {}, total examples to evaluate: {}'.format(k, len(dataset)))
    ranks = [val['rank'] == 1 for val in pred_results.values()]
    print('approximate top1 lowerbound: {}, {}/{}'.format(np.mean(ranks), np.sum(ranks), len(ranks)))
    if save_res:
        beam_res_file = os.path.join(args.filename, 'beam_result_{}.json'.format(k))
        with open(beam_res_file, 'w') as f:
            json.dump(pred_results, f, indent=4)


def eval_multi_process(args, model, device, test_dataset):
    model.eval()
    data_chunks = []
    chunk_size = len(test_dataset.processed_data_files) // args.num_process + 1
    print('total examples to evaluate:', len(test_dataset.processed_data_files))
    for i in range(0, len(test_dataset.processed_data_files), chunk_size):
        data_chunks.append(test_dataset.processed_data_files[i:i + chunk_size])
        print('chunk size:', len(data_chunks[-1]))

    mp.set_start_method('spawn')
    model.share_memory()
    processes = []
    results = []
    for k, data_files in enumerate(data_chunks):
        test_dataset.processed_data_files = data_files
        res_file = os.path.join(args.filename, 'beam_result_{}.json'.format(k))
        results.append(res_file)
        p = mp.Process(
            target=eval_decoding,
            args=(args, model, device, test_dataset, True, k)
        )
        # We first train the model across `num_process` processes
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    pred_results = {}
    for res_file in results:
        pred_results.update(json.load(open(res_file)))
    with open(os.path.join(args.filename, 'beam_result_{}.json'.format(args.eval_split)), 'w') as f:
        json.dump(pred_results, f, indent=4)


def train_multiprocess(rank, args, model, device, train_dataset, valid_dataset):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    epochs = args.epochs // args.num_process
    output_model_file = os.path.join(args.filename, 'model.pt')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    valid_dataset_ = copy.deepcopy(valid_dataset)
    valid_dataset_.processed_data_files = valid_dataset_.processed_data_files_valid
    val_loader = DataLoader(valid_dataset_, batch_size=args.batch_size, shuffle=False)
    for epoch in range(1, epochs + 1):
        print("====rank and epoch: ", rank, epoch)
        train_res = train(args, model, device, train_loader, optimizer)
        log = "rank: %d epoch: %d train_loss: %f loss_prod: %f loss_react: %f " \
              "prod_pred_acc_max: %f react_pred_acc_each: %f react_pred_acc: %f" % (rank, epoch, *train_res)
        print(log)
        scheduler.step()
        if rank == 0:
            torch.save(model.state_dict(), output_model_file)
        print("====evaluation")
        val_res = train(args, model, device, val_loader, train=False)
        log = "rank: %d epoch: %d val_loss: %f loss_prod: %f loss_react: %f " \
              "prod_pred_acc_max: %f react_pred_acc_each: %f react_pred_acc: %f" % (rank, epoch, *val_res)
        print(log)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=80,
                        help='number of epochs to train (default: 80)')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=6,
                        help='number of GNN message passing layers (default: 6).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.2,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="attention",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="concat",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default='data/USPTO50K',
                        help='root directory of dataset.')
    parser.add_argument('--atom_feat_dim', type=int, default=45,
                        help="atom feature dimension.")
    parser.add_argument('--bond_feat_dim', type=int, default=12,
                        help="bond feature dimension.")
    parser.add_argument('--onehot_center', action='store_true', default=False,
                        help='reaction center encoding: onehot or subgraph')
    parser.add_argument('--center_loss_type', type=str, default='ce',
                        help='loss type (bce or ce) for reaction center prediction')
    parser.add_argument('--typed', action='store_true', default=False,
                        help='if given reaction types')
    parser.add_argument('--test_only', action='store_true', default=False,
                        help='only evaluate on test data')
    parser.add_argument('--multiprocess', action='store_true', default=False,
                        help='train a model with multi process')
    parser.add_argument('--num_process', type=int, default=4,
                        help='number of processes for multi-process training')
    parser.add_argument('--input_model_file', type=str, default='',
                        help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default='uspto50k',
                        help='output filename')
    parser.add_argument('--runseed', type=int, default=0,
                        help="Seed for minibatch selection, random initialization.")
    parser.add_argument('--eval_split', type=str, default='test',
                        help='evaluation test/valid/train dataset')
    parser.add_argument('--beam_size', type=int, default=50,
                        help='beam search size for rnn decoding')
    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    # set up dataset
    if args.test_only:
        assert args.eval_split in ['test', 'valid', 'train']
        test_dataset = MoleculeDataset(args.dataset, split=args.eval_split, load_mol=True)
        prod_word_size = len(test_dataset.prod_smarts_fp_list)
        react_word_size = len(test_dataset.react_smarts_list)
    else:
        train_dataset = MoleculeDataset(args.dataset, split='train', load_mol=True)
        valid_dataset = MoleculeDataset(args.dataset, split='valid', load_mol=False)
        prod_word_size = len(train_dataset.prod_smarts_fp_list)
        react_word_size = len(train_dataset.react_smarts_list)

    if args.typed:
        args.atom_feat_dim += 10
        args.filename += '_typed'

    # set up model
    model = GNN_graphpred(args.num_layer, args.emb_dim, args.atom_feat_dim, args.bond_feat_dim,
                          args.center_loss_type, 0, prod_word_size, react_word_size, JK=args.JK,
                          drop_ratio=args.dropout_ratio, graph_pooling=args.graph_pooling)
    del model.gnn_diff
    del model.scoring
    model.to(device)

    dataset = os.path.basename(args.dataset)
    args.filename = os.path.join('logs', dataset, args.filename)
    os.makedirs(args.filename, exist_ok=True)
    if not args.input_model_file == "":
        input_model_file = os.path.join(args.filename, args.input_model_file)
        model.from_pretrained(input_model_file, args.device)
        print("load model from:", input_model_file)

    if args.test_only:
        print("evaluate on test data only")
        if args.multiprocess:
            eval_multi_process(args, model, device, test_dataset)
        else:
            eval_decoding(args, model, device, test_dataset)
        exit(1)

    if args.multiprocess:
        mp.set_start_method('spawn', force=True)
        model.share_memory()  # gradients are allocated lazily, so they are not shared here
        processes = []
        output_model_files = []
        for rank in range(args.num_process):
            output_model_files.append(os.path.join(args.filename, 'model_{}.pt'.format(rank)))
            p = mp.Process(
                target=train_multiprocess,
                args=(rank, args, model, device, train_dataset, valid_dataset)
            )
            # We first train the model across `num_process` processes
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
            ret = train(args, model, device, train_loader, optimizer)
            print(ret)
            scheduler.step()
            torch.save(model.state_dict(), output_model_file)


if __name__ == "__main__":
    main()
