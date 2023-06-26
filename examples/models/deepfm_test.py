#!/usr/bin/python
# -*- coding: UTF-8 -*-
#
# `Damo-Embedding` - 'c++ tool for sparse parameter server'
# Copyright (C) 2019 - present timepi <timepi123@gmail.com>
# `Damo-Embedding` is provided under: GNU Affero General Public License
# (AGPL3.0) https:#www.gnu.org/licenses/agpl-3.0.html unless stated otherwise.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#

import time
import torch
from deepfm import DeepFM
from embedding import Storage
from sklearn.metrics import roc_auc_score
from data_prepare import process as data_process


def process(train_loader, valid_loader, epochs=1):
    model = DeepFM(8, 39)
    loss_fcn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    best_auc = 0.0
    for _ in range(epochs):
        model.train()
        train_loss_sum = 0.0
        start_time = time.time()
        for idx, x in enumerate(train_loader):
            features, label = x[0], x[1]
            pred = model(features).view(-1)
            loss = loss_fcn(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.cpu().item()
            if (idx + 1) % 10 == 0 or (idx + 1) == len(train_loader):
                print(
                    "Epoch {:04d} | Step {:04d} / {} | Loss {:.4f} | Time {:.4f}".format(
                        _ + 1,
                        idx + 1,
                        len(train_loader),
                        train_loss_sum / (idx + 1),
                        time.time() - start_time,
                    )
                )
        model.eval()
        with torch.no_grad():
            valid_labels, valid_preds = [], []
            for idx, x in enumerate(valid_loader):
                features, label = x[0], x[1]
                pred = model(features).reshape(-1).numpy().tolist()
                valid_preds.extend(pred)
                valid_labels.extend(label.numpy().tolist())
        cur_auc = roc_auc_score(valid_labels, valid_preds)
        if cur_auc > best_auc:
            best_auc = cur_auc
            # torch.save(model.state_dict(), "data/deepfm_best.pth")
        print("Current AUC: %.6f, Best AUC: %.6f\n" % (cur_auc, best_auc))
    Storage.checkpoint("./checkpoint")


if __name__ == "__main__":
    train_loader, valid_loader = data_process("train.txt")
    process(train_loader, valid_loader, 1)