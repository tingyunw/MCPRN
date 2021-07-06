#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import networkx as nx
import numpy as np
import torch


def data_masks(all_usr_pois, item_tail):
    #seq padding:post
    #us_pois:padding seq, us_masks:padding item or not, len_max: length of max seq
    us_lens = [len(upois) for upois in all_usr_pois]
    len_mean = np.mean(us_lens)
    len_max = max(us_lens)
    # len_max = 10
    if len_max >= 20:
        len_max = 20
    #length = 10
    us_pois = [item_tail * (len_max - le) + upois if le < len_max else upois[-len_max:] for upois, le in zip(all_usr_pois, us_lens)]
    # us_msks = [[0] * (len_max - le) + [1] * le  for le in us_lens]
    us_msks = [[0] * (len_max - le) + [1] * le if le < len_max else [1] * len_max for le in us_lens]
    return us_pois, us_msks, len_max


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data():
    def __init__(self, data, n_node, shuffle=False, graph=None, n_negative=199):
        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph
        self.n_node = n_node
        self.all_items = np.arange(1,self.n_node)
        self.n_negative = n_negative


    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i):
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        neg_sample = []
        for t in targets:
            # t_ = self.all_items.tolist()
            # t_.remove(t)
            neg_sample.append(np.random.choice(np.setdiff1d(self.all_items,t),self.n_negative))
        targets = np.concatenate((targets.reshape(-1,1), np.array(neg_sample)),1)
        # items, n_node, A, alias_inputs = [], [], [], []
        # for u_input in inputs:
        #     n_node.append(len(np.unique(u_input)))
        # max_n_node = np.max(n_node) #max num of session node in batch
        # for u_input in inputs:
        #     node = np.unique(u_input)
        #     #batch中 相鄰矩陣size相同 不足補0
        #     items.append(node.tolist() + (max_n_node - len(node)) * [0])
        #     u_A = np.zeros((max_n_node, max_n_node))
        #     for i in np.arange(len(u_input) - 1):
        #         if u_input[i + 1] == 0:
        #             break
        #         u = np.where(node == u_input[i])[0][0]
        #         v = np.where(node == u_input[i + 1])[0][0]
        #         u_A[u][v] = 1
        #     u_sum_in = np.sum(u_A, 0)
        #     u_sum_in[np.where(u_sum_in == 0)] = 1
        #     u_A_in = np.divide(u_A, u_sum_in)
        #     u_sum_out = np.sum(u_A, 1)
        #     u_sum_out[np.where(u_sum_out == 0)] = 1
        #     u_A_out = np.divide(u_A.transpose(), u_sum_out)
        #     u_A = np.concatenate([u_A_in, u_A_out]).transpose()
        #     A.append(u_A)
        #     #alias_inputs:input items對應A中的位置
        #     alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        # return alias_inputs, A, items, mask, targets
        return inputs, mask, targets
