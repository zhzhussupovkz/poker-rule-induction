#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import os
import time
from pandas import DataFrame, read_csv

from sklearn import cross_validation, grid_search
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import pylab as pl
import matplotlib.pyplot as plt

dirs = ['data_plot', 'test_plot']

def start():
    for i in dirs:
        if not os.path.exists(i):
            os.makedirs(i)

def get_analyze_data():
    print "Get analyze data..."
    data = read_csv("./train.csv")
    data['id'] = range(1, len(data)+1)
    return data

def plot_data():
    data = get_analyze_data()
    headers = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4','S5', 'C5']
    for k in headers:
        print "Plot %s..." % k
        f = plt.figure(figsize = (8, 6))
        p = data.pivot_table('id', k, 'hand', 'count').plot(kind = 'barh', stacked = True, title = '%s' % k, ax = f.gca())
        f.savefig('./%s/%s_hand.png' % (dirs[0], k))

def get_test_data():
    print "Get test data..."
    data = read_csv("./test.csv")
    result = DataFrame(data['id'])

    data = data.drop(['id'], axis = 1)
    return (data, result)

def cross_validation_test():
    data = get_analyze_data()
    target = data["hand"]
    train = data.drop(["id"], axis = 1)
    kfold = 5
    cross_val_test = {}

    print "Cross validation test..."
    model_rfc = RandomForestClassifier(n_estimators = 100)
    model_knc = KNeighborsClassifier(n_neighbors = 15)
    model_lr = LogisticRegression(penalty='l1', tol=0.01)

    scores = cross_validation.cross_val_score(model_rfc, train, target, cv = kfold)
    cross_val_test['RFC'] = scores.mean()

    scores = cross_validation.cross_val_score(model_knc, train, target, cv = kfold)
    cross_val_test['KNC'] = scores.mean()

    scores = cross_validation.cross_val_score(model_lr, train, target, cv = kfold)
    cross_val_test['LR'] = scores.mean()

    f = plt.figure(figsize = (8, 6))
    p = DataFrame.from_dict(data = cross_val_test, orient='index').plot(kind='barh', legend=False, ax = f.gca())
    f.savefig('./%s/cross_validation_test.png' % dirs[1])

    for k,v in cross_val_test.iteritems():
        print "%s : %s" % (k,str(v))

def grid_search_test():
    data = get_analyze_data()
    target = data['hand']
    train = data.drop(['id', 'hand'], axis = 1)

    print "Grid search test..."
    model_rfc = RandomForestClassifier(n_estimators = 256)
    # params = {"n_estimators" : [100, 125, 225, 250]}
    params = {"criterion" : ('entropy', 'gini')}
    clf = grid_search.GridSearchCV(model_rfc, params)
    clf.fit(train, target)

    print (clf.best_score_)
    print (clf.best_estimator_.n_estimators)
    print (clf.best_estimator_.criterion)

def go():
    data = get_analyze_data()

    # best result - RandomForestClassifier
    model_rfc = RandomForestClassifier(n_estimators=512, n_jobs=-1)

    print "Go!!!"
    print "RFC..."

    test, result = get_test_data()
    target = data['hand']
    train = data.drop(['id', 'hand'], axis = 1)

    print "..."
    model_rfc.fit(train, target)
    result.insert(1,'hand', model_rfc.predict(test))
    result.to_csv('./test_rfc_256_2.csv', index=False)

def go_gbc():
    data = get_analyze_data()

    model_gbm = GradientBoostingClassifier(n_estimators=1024)

    print "Go!!!"
    print "GBM..."

    test, result = get_test_data()
    target = data['hand']
    train = data.drop(['id', 'hand'], axis = 1)

    print "..."
    model_gbm.fit(train, target)
    result.insert(1,'hand', model_gbm.predict(test))
    result.to_csv('./test_gbm_1024.csv', index=False)

start = time.clock()
# start()
# plot_data()
# cross_validation_test()
# go()
go_gbc()
# grid_search_test()
end = time.clock()
print "Time: %s" % str(end-start)
