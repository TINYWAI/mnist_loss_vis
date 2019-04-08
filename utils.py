import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

def visualize(feat, labels, epoch, path):
    # plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.clf()
    for i in range(10):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc = 'upper right')
    plt.text(0, 0, "epoch=%d" % epoch)
    save_file_path = os.path.join(path, 'epoch=%d.jpg' % epoch)
    plt.savefig(save_file_path)
    plt.close()


def visualize2(train_feat, train_label, test_feat, test_label, epoch, path, train_acc=None, test_acc=None):
    # plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    feats = [train_feat, test_feat]
    labels = [train_label, test_label]
    if train_acc is not None and test_acc is not None:
        accs = [train_acc, test_acc]
        titles = ['train_acc: ', 'test_acc: ']
    else:
        accs = None
        titles = None
    plt.figure(figsize=(9, 4))
    for i in range(2):
        feat = feats[i]
        label = labels[i]
        plt.subplot(1, 2, i+1)
        for j in range(10):
            plt.plot(feat[label == j, 0], feat[label == j, 1], '.', c=c[j])
        if accs is not None:
            plt.title(titles[i]+'%.3f%%'%accs[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc = 'upper right')
    save_file_path = os.path.join(path, 'epoch=%d.jpg'%epoch)
    plt.savefig(save_file_path)
    plt.close()

def visualize_3D(feat, labels, epoch, path):
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    for i in range(10):
        ax1.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc = 'upper right')
    plt.text(0, 0, "epoch=%d" % epoch)
    save_file_path = os.path.join(path, 'epoch=%d.jpg' % epoch)
    plt.savefig(save_file_path)
    plt.close()


def visualize2_3D(train_feat, train_label, test_feat, test_label, epoch, path, train_acc=None, test_acc=None):
    # plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    feats = [train_feat, test_feat]
    labels = [train_label, test_label]
    if train_acc is not None and test_acc is not None:
        accs = [train_acc, test_acc]
        titles = ['train_acc: ', 'test_acc: ']
    else:
        accs = None
        titles = None
    fig = plt.figure(figsize=(9, 4))
    for i in range(2):
        feat = feats[i]
        label = labels[i]
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        for j in range(10):
            ax.scatter(feat[label == j, 0], feat[label == j, 1], feat[label == j, 2], '.', c=c[j])
        if accs is not None:
            plt.title(titles[i]+'%.3f%%'%accs[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc = 'upper right')
    save_file_path = os.path.join(path, 'epoch=%d.jpg'%epoch)
    plt.savefig(save_file_path)
    plt.close()


if __name__ == '__main__':

    #
    # # 创建 1 张画布
    # fig = plt.figure()
    #
    # ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    #
    # data = np.arange(24).reshape((8, 3))
    # # 绘制第 1 张图
    # for i in range(8):
    #     ax1.scatter(data[i, 0], data[i, 1], data[i, 2])
    #
    # # 显示图
    # plt.show()

    def visualize_3D_test(feat, labels):
        c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
             '#ff00ff', '#990000', '#999900', '#009900', '#009999']
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1, projection='3d')
        for i in range(10):
            ax1.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
        plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
        plt.show()


    def visualize2_3D_test(train_feat, train_label, test_feat, test_label):
        # plt.ion()
        c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
             '#ff00ff', '#990000', '#999900', '#009900', '#009999']
        feats = [train_feat, test_feat]
        labels = [train_label, test_label]
        fig = plt.figure(figsize=(9, 4))
        for i in range(2):
            feat = feats[i]
            label = labels[i]
            ax = fig.add_subplot(1, 2, i + 1, projection='3d')
            for j in range(10):
                ax.scatter(feat[label == j, 0], feat[label == j, 1], feat[label == j, 2], '.', c=c[j])
            plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
        plt.show()


    data = np.arange(24).reshape((8, 3))
    label = np.arange(8)

    visualize2_3D_test(data, label,data, label)