import numpy as np

# 定义训练样本集
D = [
    (np.array([0.2, 0.7]), 1),
    (np.array([0.3, 0.3]), 1),
    (np.array([0.4, 0.5]), 1),
    (np.array([0.6, 0.5]), 1),
    (np.array([0.1, 0.4]), 1),
    (np.array([0.4, 0.6]), -1),
    (np.array([0.6, 0.2]), -1),
    (np.array([0.7, 0.4]), -1),
    (np.array([0.8, 0.6]), -1),
    (np.array([0.7, 0.5]), -1),
]

# 初始化权重
weights = np.zeros(3)  # 包括偏置项
max_iterations = 20
best_weights = weights.copy()
best_error_count = float('inf')
results = []

# Pocket算法实现
for iteration in range(max_iterations):
    error_count = 0
    misclassified = []

    # 计算当前模型对每个样本的分类
    for x, y in D:
        # 添加偏置项
        x_with_bias = np.insert(x, 0, 1)
        if np.sign(np.dot(weights, x_with_bias)) != y:
            error_count += 1
            misclassified.append((x, y))

    # 记录当前迭代结果
    results.append((iteration, weights.copy(), error_count))

    # 输出当前迭代的结果和到目前为止的最佳结果
    print(f"Iteration {iteration + 1}: Current weights = {weights}, Error count = {error_count}")
    print(f"Best weights so far = {best_weights}, Best error count so far = {best_error_count}\n")

    # 如果当前的错误分类数量少于最佳错误分类数量，更新最佳权重
    if error_count < best_error_count:
        best_error_count = error_count
        best_weights = weights.copy()

    # 更新权重
    if misclassified:
        # 随机选择一个错误分类的样本
        x_misclass, y_misclass = misclassified[np.random.choice(len(misclassified))]
        x_misclass_with_bias = np.insert(x_misclass, 0, 1)
        weights += y_misclass * x_misclass_with_bias

# 输出最终的最佳结果
print(f"Final best weights after {max_iterations} iterations: {best_weights}")
print(f"Final best error count: {best_error_count}")
