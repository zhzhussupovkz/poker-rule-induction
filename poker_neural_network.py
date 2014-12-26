#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import time
import csv
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

def get_data():
    data = []
    with open("./train.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def get_ds():
    ds = SupervisedDataSet(10, 1)
    data = get_data()
    for k in data:
        ds.addSample((k.get('S1'), k.get('C1'), \
            k.get('S2'), k.get('C2'), \
            k.get('S3'), k.get('C3'), \
            k.get('S4'), k.get('C4'), \
            k.get('S5'), k.get('C5')), k.get('hand'))
    return ds

def train_neural_network():
    ds = get_ds()
    net = buildNetwork(10,3,1, bias=True)
    trainer = BackpropTrainer(net, ds)
    return trainer.trainUntilConvergence()

t = train_neural_network()
print t