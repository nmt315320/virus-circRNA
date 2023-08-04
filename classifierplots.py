#!/usr/bin/env python3
# -*- coding=utf-8 -*-

import matplotlib
matplotlib.use('Agg')

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE


def make_meshgrid(x, y, h=.001):

    x_min, x_max = x.min() - 0.3, x.max() + 0.3
    y_min, y_max = y.min() - 0.3, y.max() + 0.3
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier."""
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, **params)


def scatter_visual(pltdata, size):
    """对t-SNE降维的数据进行可视化"""
    tsne_df = pd.DataFrame(
        data=pltdata, columns=("Dimension1", "Dimension2", "label"))
    p1 = tsne_df[(tsne_df.label == 1)]
    p2 = tsne_df[(tsne_df.label == 0)]
    x1 = p1.values[:, 0]
    y1 = p1.values[:, 1]
    x2 = p2.values[:, 0]
    y2 = p2.values[:, 1]

    # 绘制散点图
    plt.plot(x1, y1, 'o', color="#3dbde2", label='positive', markersize=size)
    plt.plot(x2, y2, 'o', color="#b41f87", label='negative', markersize=size)
    plt.xlabel('Dimension1', fontsize=9)
    plt.ylabel('Dimension2', fontsize=9)
    plt.margins(0.3)
    plt.legend(loc="upper right")


def tsne_data(rawdata):
    """对输入数据进行处理降维并归一化返回降维结果"""
    modle = TSNE(n_components=2, random_state=0)

    fea_data = rawdata.drop(columns=['class'])  # 取出所有特征向量用于降维
    redu_fea = modle.fit_transform(fea_data)  # 将数据降到2维进行后期的可视化处理
    scaler = MinMaxScaler()
    norm_data = scaler.fit_transform(redu_fea)  # 对降维数据进行归一化

    return norm_data, rawdata['class']

    return grid_search.best_estimator_


def getbest(clf, para, X, y):
    """get the best parameter for classifier by GridsearchCV"""
    grid_search = GridSearchCV(clf, para, cv=5)

    # 训练寻找最优参数
    grid_search.fit(X, y)

    print("Best parameters:{}".format(grid_search.best_params_))
    print("Best score on train set:{:.2f}".format(grid_search.best_score_))

    return grid_search.best_estimator_


def main():
    parser = argparse.ArgumentParser(
        description=
        "Use tsne to reduce the dimensions and display the data, you can draw the decision boundary of various classifier"
    )
    parser.add_argument(
        '-i',
        '--infile',
        help=
        "The input file should be csv format, and multiple file should be separated by commas",
        required=True)
    parser.add_argument(
        '-o', '--outfile', help='The name of output picture', required=True)
    parser.add_argument(
        '-d',
        '--dpi',
        help='The dpi of output picture, default is 300dpi',
        default=300)
    parser.add_argument(
        '-p', '--pointsize', help='The size of point', type=float, default=3)
    parser.add_argument(
        '-f',
        '--classifier',
        help=
        'The classifier decides the decision boundary, default is 0:0=SVM, 1=logisticRedgression, 2=NaiveByes, 3=QDA',
        type=int,
        default=0)
    args = parser.parse_args()

    classifiers = [
        SVC(kernel='rbf', probability=True),
        LogisticRegressionCV(cv=5, random_state=0, multi_class='multinomial'),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()
    ]
    parameters = [{
        'gamma': [2**x for x in range(-5, 15, 2)],
        "C": [2**x for x in range(-15, 3, 2)]
    }]

    csvdata = pd.read_csv(args.infile)
    X, y = tsne_data(csvdata)

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    # 对可以优化的参数进行调优
    f = args.classifier
    if f == 0:
        clf = getbest(classifiers[f], parameters[f], X, y)
    else:
        clf = classifiers[f].fit(X, y)
    plot_contours(clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.5)
    scatter_visual(np.vstack((X.T, y.T)).T, args.pointsize)

    # 调节子图之间的距离
    plt.subplots_adjust(
        top=0.92, bottom=0.20, left=0.15, right=0.95, hspace=0.20, wspace=0.20)
    plt.savefig(args.outfile, dpi=args.dpi)


if __name__ == "__main__":
    main()