#!/usr/bin/env python3
# -*- coding=utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import math

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="将csv格式数据的部分维度的特征进行展示")
    parser.add_argument("-i", "--input", help="输入文件名称（绝对路径/相对路径）")
    parser.add_argument('-o', "--output", help="输出文件名称（相对路径/绝对路径）")
    parser.add_argument('-n', "--number", help="需要展示的维度的个数(int）", type=int)
    parser.add_argument(
        '-c', "--canvas", help="画布尺寸大小,默认为25", type=int, default=25)
    parser.add_argument('-f', "--fontsize", help="字体大小", type=float, default=1)
    parser.add_argument('-w', "--wspace", help="子画布宽距", type=float, default=0.5)
    args = parser.parse_args()

    rawdata = pd.read_csv(args.input)
    header = list(rawdata.columns)
    header.remove("class")
    # x = np.array(rawdata.iloc[:, :-1])
    # y = np.array(rawdata.iloc[:, -1])
    # columns = list(rawdata.columns)

    # plt.figure(figsize=(args.canvas, args.canvas))
    plt.tight_layout()
    # plt.subplots_adjust(wspace=args.wspace)
    plt_count = math.ceil(math.sqrt(args.number))
    for i in range(2, args.number + 2):
        # sns.set(style='white', font_scale=args.fontsize)
        plt.subplot(plt_count, plt_count, i - 1)
        ax = sns.violinplot(x="class", y=header[i - 2], data=rawdata,palette=["#b41f87", "#3dbde2"], inner=None)
        ax = sns.swarmplot(
            x="class",
            y=header[i - 2],
            size=3,
            data=rawdata,
            color='w',
            alpha=.5)
        plt.xlabel('')
        plt.ylabel('')
        # plt.legend(fontsize=12, framealpha=0.5, loc="upper right")
        plt.tick_params(labelsize=16)
        
        
    plt.savefig(args.output, dpi=300)
    plt.show()