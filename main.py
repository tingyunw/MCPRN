#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import argparse
import pickle
import time
from utils import Data, split_validation
from model import *
import copy
import torch
import os
from datetime import datetime as dt

torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='jdata', help='dataset name: diginetica/yoochoose/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--train_batchSize', type=int, default=50, help='train input batch size')
parser.add_argument('--test_batchSize', type=int, default=50, help='test input batch size')
parser.add_argument('--hiddenSize', type=int, default=128, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--n_interest', type=int, default=3, help='the number of interests')
parser.add_argument('--loss', default='PCE')
parser.add_argument('--debug', default=False)
opt = parser.parse_args()
print(opt)

today = dt.today()

def main():
    train_data = pickle.load(open('./datasets/' + opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('./datasets/' + opt.dataset + '/test.txt', 'rb'))

    if opt.dataset == 'diginetica':
        n_node = 43098
        # n_node = 43039+1
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    elif opt.dataset == 'yoochoose':
        n_node = 8937
    elif opt.dataset == 'yoochoose_buy':
        n_node = 7288
    elif opt.dataset == 'jdata':
        n_node = 44749
    else:
        n_node = 310

    if opt.debug:
        train_data = ([train_data[0][:1000], train_data[1][:1000]])
        test_data = ([test_data[0][:1000], test_data[1][:1000]])
    train_data = Data(train_data, shuffle=True, n_node = n_node)
    test_data = Data(test_data, shuffle=False ,n_node = n_node)

    model = trans_to_cuda(SessionGraph(opt, n_node, opt.n_interest))

    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)

        previous_model = copy.deepcopy(model)
        previous_model.eval()

        hit, mrr = train_test(model, train_data, test_data)

        if opt.debug:
            print(torch.all(torch.eq(previous_model.gnn[0].b_hh, model.gnn[0].b_hh)))
            print(torch.all(torch.eq(previous_model.gnn[0].w_ih, model.gnn[0].w_ih)))
            # print(torch.all(torch.eq(previous_model.gnn.b_hh, model.gnn.b_hh)))
            # print(torch.all(torch.eq(previous_model.gnn.w_ih, model.gnn.w_ih)))
            print(torch.all(torch.eq(previous_model.embedding.weight, model.embedding.weight)))
            print(torch.all(torch.eq(previous_model.w_interest.weight, model.w_interest.weight)))
            print(len(model.optimizer.param_groups[0]['params']))  # 算optimizer吃到幾個參數

        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMRR@20:\t%.4f\tEpoch:\t%d,\t%d'% (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        model_file = os.getcwd() + '/models/' + opt.dataset + '_%s_%s_%s_I1.pt' % (str(today.year), str(today.month), str(today.day))
        torch.save(model.state_dict(), model_file)
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))



if __name__ == '__main__':
    main()
