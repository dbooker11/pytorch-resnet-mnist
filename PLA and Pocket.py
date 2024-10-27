import numpy as np
import matplotlib.pyplot as plt
from time import time


# 产生数据集
def generate_datasets():
    np.random.seed(42)  # 为了结果可复现
    m1 = np.array([1, 0])
    m2 = np.array([0, 1])
    cov = np.eye(2)

    # 生成200个样本
    X1 = np.random.multivariate_normal(m1, cov, 200)
    X2 = np.random.multivariate_normal(m2, cov, 200)

    # 标签
    y1 = np.ones(200)
    y2 = -np.ones(200)

    # 合并数据集
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))

    # 划分训练集和测试集
    train_size = int(0.8 * len(X))
    indices = np.random.permutation(len(X))
    X_train, X_test = X[indices[:train_size]], X[indices[train_size:]]
    y_train, y_test = y[indices[:train_size]], y[indices[train_size:]]

    return X_train, y_train, X_test, y_test


# PLA算法实现
class PLA:
    def __init__(self, max_iterations=1000):
        self.max_iterations = max_iterations

    def fit(self, X, y):
        # 加一个偏置项
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.w = np.zeros(X.shape[1])
        for _ in range(self.max_iterations):
            for i in range(len(X)):
                if y[i] * np.dot(X[i], self.w) <= 0:
                    self.w += y[i] * X[i]

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return np.sign(np.dot(X, self.w))


# Pocket算法实现
class PocketAlgorithm:
    def __init__(self, max_iterations=1000):
        self.max_iterations = max_iterations

    def fit(self, X, y):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.w = np.zeros(X.shape[1])
        best_w = np.zeros(X.shape[1])
        best_accuracy = 0

        for _ in range(self.max_iterations):
            for i in range(len(X)):
                if y[i] * np.dot(X[i], self.w) <= 0:
                    self.w += y[i] * X[i]

            # 计算当前的准确率
            predictions = np.sign(np.dot(X, self.w))
            accuracy = np.mean(predictions == y)

            # 更新最佳权重
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_w = self.w.copy()

        self.w = best_w

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return np.sign(np.dot(X, self.w))


# 计算准确率
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


# 可视化数据集和分类面
def plot_decision_boundary(X, y, model, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Class +1', color='blue', alpha=0.6)
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], label='Class -1', color='red', alpha=0.6)

    # 绘制分类面
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0], colors='green')

    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid()
    plt.show()


# 主程序
X_train, y_train, X_test, y_test = generate_datasets()

# PLA算法
pla = PLA()
start_time = time()
pla.fit(X_train, y_train)
train_time_pla = time() - start_time

# 测试PLA
train_pred_pla = pla.predict(X_train)
test_pred_pla = pla.predict(X_test)
accuracy_train_pla = accuracy(y_train, train_pred_pla)
accuracy_test_pla = accuracy(y_test, test_pred_pla)

# Pocket算法
pocket = PocketAlgorithm()
start_time = time()
pocket.fit(X_train, y_train)
train_time_pocket = time() - start_time

# 测试Pocket
train_pred_pocket = pocket.predict(X_train)
test_pred_pocket = pocket.predict(X_test)
accuracy_train_pocket = accuracy(y_train, train_pred_pocket)
accuracy_test_pocket = accuracy(y_test, test_pred_pocket)

# 输出结果
print(f"PLA Train Accuracy: {accuracy_train_pla:.2f}, Test Accuracy: {accuracy_test_pla:.2f}")
print(f"PLA Training Time: {train_time_pla:.4f} seconds")
print(f"Pocket Train Accuracy: {accuracy_train_pocket:.2f}, Test Accuracy: {accuracy_test_pocket:.2f}")
print(f"Pocket Training Time: {train_time_pocket:.4f} seconds")

# 可视化结果
plot_decision_boundary(X_train, y_train, pla, 'PLA Decision Boundary')
plot_decision_boundary(X_train, y_train, pocket, 'Pocket Algorithm Decision Boundary')
