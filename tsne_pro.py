#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from sklearn.manifold import TSNE

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse


def tsne_visual(pltdata):
    """对t-SNE降维的数据进行可视化"""

    p1 = pltdata[(pltdata.label=="positive")]
    p2 = pltdata[(pltdata.label=="negative")]
    x1 = p1.values[:, 0]
    y1 = p1.values[:, 1]
    x2 = p2.values[:, 0]
    y2 = p2.values[:, 1]

    #绘制散点图
    plt.plot(x1, y1, 'o', color="#3dbde2", label='positive', markersize='3')
    plt.plot(x2, y2, 'o', color="#b41f87", label='negative', markersize='3')
    plt.xlabel('Dimension1', fontsize=9)
    plt.ylabel('Dimension2', fontsize=9)
    # left, right = plt.gca().get_xlim()
    # top, bottom = plt.gca().get_ylim()
    # plt.xlim((left*1.5, right*1.5))
    # plt.ylim((top*1.3, bottom*1.3))
    # plt.title(title, fontsize=12)
    plt.margins(0.3)
    plt.legend(loc="upper right")
    

def tsne_data(rawdata):
    """对输入数据进行处理降维并返回降维结果"""
    modle = TSNE(n_components=2, random_state=0)

    fea_data = rawdata.drop(columns=['class'])  #取出所有特征向量用于降维
    redu_fea = modle.fit_transform(fea_data) #将数据降到2维进行后期的可视化处理
    labels = rawdata['class'].replace([0, 1], ['negative', 'positive']) #将正反例数字替换成正反例标签用于收起展示
    redu_data = np.vstack((redu_fea.T, labels.T)).T #将特征向量和正反例标签整合
    tsne_df = pd.DataFrame(data=redu_data, columns=['Dimension1', 'Dimension2', "label"])
    tsne_visual(tsne_df)


def main():
    parser = argparse.ArgumentParser(description="Visualzes the features based on t-sne in a 2D feature space")
    parser.add_argument('-i', '--infile', help="The input file should be csv format, and multiple file should be separated by commas", required=True)
    parser.add_argument('-o', '--outfile', help='The name of output picture', required=True)
    parser.add_argument('-l', '--layout', help='The layout of subplots, please input row,column number, default 1,1 ', default='1,1')
    parser.add_argument('-d', '--dpi', help='The dpi of output picture, default is 300dpi', default=300)
    args = parser.parse_args()

    infile_list = args.infile.split(',') # 获得文件名列表
    row, column = [int(i) for i in args.layout.split(',')] 
    assert len(infile_list) <= row*column, "The number of inputfile to be drawn is not greater than the number of subplots of the canvas, please set '-l' parameter"
    
    # 通过循环对各个文件进行处理，使用数字索引是为了确定子画布的位置
    plt.figure(figsize=(column*4.7, row*3)) #设置画布的尺寸

    for i in range(len(infile_list)):
        plt.subplot(row, column, i+1)
        csvdata = pd.read_csv(infile_list[i])
        tsne_data(csvdata)

    plt.subplots_adjust(top=0.92, bottom=0.20, left=0.15, right=0.95, hspace=0.20, wspace=0.20) # 调整子图之间的分布
    plt.savefig(args.outfile, dpi=args.dpi)    

if __name__ == "__main__":
    main()
