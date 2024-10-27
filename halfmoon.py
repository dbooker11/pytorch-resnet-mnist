import numpy as np
import matplotlib.pyplot as plt


def halfmoon(rad, width, d, n_samp):
    if n_samp % 2 != 0:
        n_samp += 1

    data = np.zeros((3, n_samp))

    a1 = np.random.random((2, n_samp // 2))  # 生成[0,1]的均匀分布随机数
    radius = (rad - width / 2) + width * a1[0, :]  # a1的第1行随机数用于确定(极坐标)半径
    theta = np.pi * a1[1, :]  # a1的第2行随机数用于确定(极坐标)弧度

    x1 = radius * np.cos(theta)  # 极坐标转直角坐标x
    y1 = radius * np.sin(theta)  # 极坐标转直角坐标y
    label1 = np.ones((1, len(x1)))  # label for Class 1

    a2 = np.random.random((2, n_samp // 2))  # 生成[0,1]的均匀分布随机数
    radius = (rad - width / 2) + width * a2[0, :]  # a2的第1行随机数用于确定(极坐标)半径
    theta = np.pi * a2[1, :]  # a2的第2行随机数用于确定(极坐标)弧度

    x2 = radius * np.cos(-theta) + rad  # 负例的角度取反，x坐标右移一个半径
    y2 = radius * np.sin(-theta) - d  # 负例的角度取反，y坐标减掉两个半月的间隔
    label2 = -1 * np.ones((1, len(x2)))  # label for Class 2

    data[0, :] = np.concatenate([x1, x2])
    data[1, :] = np.concatenate([y1, y2])
    data[2, :] = np.concatenate([label1, label2], axis=1)

    return data


def halfmoon_shuffle(rad, width, d, n_samp):
    data = halfmoon(rad, width, d, n_samp)
    shuffle_seq = np.random.permutation(np.arange(n_samp))
    data_shuffle = data[:, shuffle_seq]

    return data_shuffle


if __name__ == "__main__":
    dataNum = 2000
    data = halfmoon(10, 6, -3, dataNum)
    pos_data = data[:, 0:dataNum // 2]
    neg_data = data[:, dataNum // 2:dataNum]

    plt.figure()
    plt.scatter(pos_data[0, :], pos_data[1, :], c="b", s=3)
    plt.scatter(neg_data[0, :], neg_data[1, :], c="r", s=3)
    plt.show()
