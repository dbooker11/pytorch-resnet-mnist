import numpy as np
import matplotlib.pyplot as plt

def halfmoon (rad, width, d, n_samp):
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

def initialize_parameters(num_samples):
    """初始化参数"""
    W1 = np.random.randn(num_samples, num_samples) / np.sqrt(num_samples)
    b1 = np.zeros((1, num_samples))  # 修正b1的形状
    W2 = np.random.randn(num_samples, 2) / np.sqrt(num_samples)  # 输出层到2个类
    b2 = np.zeros((1, 2))  # 修正b2的形状
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

def forward_propagation(X, parameters):
    """前向传播"""
    W1, b1, W2, b2 = parameters['W1'], parameters['b1'], parameters['W2'], parameters['b2']
    Z1 = np.dot(X, W1) + b1  # 注意转置
    A1 = np.tanh(Z1)
    Z2 = np.dot(A1, W2) + b2
    exp_scores = np.exp(Z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return Z1, A1, Z2, probs

def compute_loss_and_gradients(probs, labels, parameters, X):
    """计算损失函数及其导数"""
    num_examples = labels.shape[0]
    loss = -np.sum(np.log(probs[range(num_examples), labels])) / num_examples
    delta3 = probs
    delta3[range(num_examples), labels] -= 1
    delta3 /= num_examples  # 归一化

    W2 = parameters['W2']
    A1 = np.tanh(np.dot(X.T, parameters['W1']) + parameters['b1'])  # 重新计算A1

    dW2 = np.dot(A1.T, delta3)
    db2 = np.sum(delta3, axis=0, keepdims=True)
    delta2 = np.dot(delta3, W2.T) * (1 - np.power(A1, 2))
    dW1 = np.dot(X.T, delta2)
    db1 = np.sum(delta2, axis=0, keepdims=True)

    return loss, {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

def update_parameters(parameters, grads, learning_rate):
    """梯度下降更新参数"""
    parameters['W1'] -= learning_rate * grads['dW1']
    parameters['b1'] -= learning_rate * grads['db1']
    parameters['W2'] -= learning_rate * grads['dW2']
    parameters['b2'] -= learning_rate * grads['db2']

if __name__ == "__main__":
    # 生成数据集
    dataNum = 2  # 特征数量
    num_samples = 200
    data = halfmoon(10, 6, -3, num_samples)
    X = data[:2, :]  # 前两行作为特征
    labels = data[2, :] # 标签

    # 初始化参数
    parameters = initialize_parameters(dataNum)

    # 超参数设置
    num_passes = 20000
    learning_rate = 0.01

    # 迭代优化
    losses = []
    for i in range(num_passes):
        Z1, A1, Z2, probs = forward_propagation(X, parameters)
        loss, gradients = compute_loss_and_gradients(probs, labels, parameters, X)
        update_parameters(parameters, gradients, learning_rate)

        if i % 1000 == 0:
            print(f"Iteration {i}: Loss = {loss:.4f}")
            losses.append(loss)

    # 绘制结果
    plt.scatter(X[0, labels == 0], X[1, labels == 0], color='red', label='Class 0')
    plt.scatter(X[0, labels == 1], X[1, labels == 1], color='blue', label='Class 1')
    plt.legend()
    plt.title('Data Points')
    plt.show()