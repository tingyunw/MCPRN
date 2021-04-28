#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F

class pair_cross_entropy(nn.Module):
    def __init__(self):
        super(pair_cross_entropy, self).__init__()

    def forward(self, targets_score):
        pos = torch.log(targets_score[:, 0])
        neg = torch.sum((torch.log(1-targets_score[:, 1:])), 1)
        loss = torch.mean(-(pos+neg))
        return loss


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.randn(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.randn(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.randn(self.gate_size))
        self.b_hh = Parameter(torch.randn(self.gate_size))

    def GNNCell(self, hidden, g_item_interest):
        #單個interest跑timestep 回傳單個interest embedding
        h_t_1 = torch.randn(hidden.shape[0], hidden.shape[2]).cuda()  # h0 random
        for t in range(hidden.shape[1]):
            gi = F.linear(hidden[:,t,:], self.w_ih, self.b_ih)
            gh = F.linear(h_t_1, self.w_hh, self.b_hh)
            i_r, i_i, i_n = gi.chunk(3,1)
            h_r, h_i, h_n = gh.chunk(3,1)
            resetgate = torch.sigmoid(i_r + h_r) #(?,hidden_size)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            g = g_item_interest[:,t].view(-1,1)
            concentrgate = (torch.gt(g,0.01).to(torch.float32))*g*inputgate
            hy = (1-concentrgate)*h_t_1 + concentrgate*newgate
            h_t_1 = hy
        return hy


    def forward(self, hidden, g_item_interest):
        #跑單個interest GRU
        interest_emb = self.GNNCell(hidden, g_item_interest)
        return interest_emb


class SessionGraph(Module):
    def __init__(self, opt, n_node, n_interest):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.train_batch_size = opt.train_batchSize
        self.test_batch_size = opt.test_batchSize
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        gnns = [trans_to_cuda(GNN(self.hidden_size)) for i in range(n_interest)]
        self.gnn = nn.ModuleList(gnns)
        self.n_interest = n_interest
        self.w_interest = nn.Linear(self.hidden_size, self.n_interest, bias=False)
        self.loss_type = opt.loss
        if self.loss_type == 'BCE':
            pos_w = torch.tensor([10]+[1]*10).to(torch.float32)
            self.loss_function = nn.CrossEntropyLoss(weight=pos_w)
        else:
            self.loss_function = pair_cross_entropy()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, all_interest_emb, targets, train):
        if train==True:
            targets_emb = self.embedding(torch.tensor(targets).to(torch.int64).cuda())
            g_item_interest = self.item_weight_for_interest(targets_emb)
            context_emb = torch.bmm(g_item_interest, all_interest_emb)
            targets_score = torch.einsum('ijk,ijk->ij', targets_emb, context_emb)
            if not self.loss_type == 'BCE':
                targets_score = 1/(1+torch.exp(-targets_score))  # BCE內建
            return targets_score
        else:
            all_items_emb = self.embedding.weight[1:]
            g_item_interest = self.item_weight_for_interest(all_items_emb)

            context_emb = g_item_interest.matmul(all_interest_emb)  # 和下面兩行一樣
            # g_item_interest = g_item_interest.repeat(all_interest_emb.shape[0], 1).view(all_interest_emb.shape[0], g_item_interest.shape[0], g_item_interest.shape[1])
            # context_emb2 = torch.bmm(g_item_interest, all_interest_emb)

            targets_score = torch.sum(all_items_emb*context_emb, dim=2)  # 與下行同
            # targets_score = torch.einsum('jk,ijk->ij', all_items_emb, context_emb)
            targets_score = 1 / (1 + torch.exp(-targets_score))
            return targets_score

    def item_weight_for_interest(self,hidden):
        '''
        算interest和item的相關度
        :param hidden: item_emb
        :return:
        '''
        alpha_item_interest = self.w_interest(hidden)  # interest和item的相關度, (?, seq, 3)
        if len(alpha_item_interest.shape) == 3:
            g_item_interest = F.softmax(alpha_item_interest / 0.1, dim=2)
        else:
            g_item_interest = F.softmax(alpha_item_interest / 0.1, dim=1)
        # 上方if-else的才是正確的, fixme 下方的不管在train/ test都是錯的
        # g_item_interest2 = torch.exp(alpha_item_interest/0.1)/torch.unsqueeze(torch.sum(torch.exp(alpha_item_interest/0.1), len(alpha_item_interest.shape)-1),len(alpha_item_interest.shape)-1)
        # print(torch.all(torch.eq(g_item_interest, g_item_interest2)))
        return g_item_interest

    def forward(self, inputs):
        all_interest_emb = []
        hidden = self.embedding(inputs).cuda()
        g_item_interest = self.item_weight_for_interest(hidden)
        g_item_interest_ = g_item_interest.cpu().detach().numpy()
        #跑m個interst各自的GRU
        for i in range(self.n_interest):
            i_interest_emb = self.gnn[i](hidden, g_item_interest[:,:,i])
            all_interest_emb.append(i_interest_emb)
        all_interest_emb = torch.cat(all_interest_emb,dim=1)

        all_interest_emb = all_interest_emb.view(-1,self.n_interest,self.hidden_size)
        return all_interest_emb


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data, train=True):
    items, mask, targets = data.get_slice(i)  #items shape:(100,69)
    items = trans_to_cuda(torch.Tensor(items).long())
    #mask = trans_to_cuda(torch.Tensor(mask).long())
    all_interest_emb_ = model(items)
    #get = lambda i: hidden[i][alias_inputs[i]]
    #seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    return targets[:,0], model.compute_scores(all_interest_emb_, targets, train)


def train_test(model, train_data, test_data):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.train_batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        # targets含1 positive和n negative samples
        targets, scores = forward(model, i, train_data)
        if model.loss_type == 'BCE':
            targets = trans_to_cuda(torch.Tensor([0]*scores.shape[0]).long())
            loss = model.loss_function(scores, targets)
        else:
            # targets = trans_to_cuda(torch.Tensor(targets).long())
            loss = model.loss_function(scores)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 50+1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
        # print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.test_batch_size)
    for i in slices:
        targets, scores = forward(model, i, test_data, train=False)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target in zip(sub_scores, targets):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr
