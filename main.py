import random
import json
import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.api import layers, optimizers
from keras.src.callbacks import EarlyStopping
from keras.src.optimizers import Adam
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt


samples = []

with open("./original_data.json", "r", encoding="utf-8") as f:
    final_data = json.load(f)
    for i in range(1, 41):
        samples.append(final_data[str(i)])

# 为nan赋所有样本对应feature的平均值
feature_avg = np.zeros(6)
feature_sums = np.zeros(6)
feature_counts = np.zeros(6)
for sample in samples:
        for i, value in enumerate(sample["feature"]):
            if value != "nan":  # 仅累加有效值
                feature_sums[i] += float(value)
                feature_counts[i] += 1
                feature_avg = feature_sums / np.where(feature_counts == 0, 1, feature_counts)

for sample in samples:
    for i, value in enumerate(sample["feature"]):
        if value == "nan":
            sample["feature"][i] = feature_avg[i]
# 为nan赋所有样本对应drug的平均值
drug_avg = np.zeros(14)
drug_sums = np.zeros(14)
drug_counts = np.zeros(14)
for sample in samples:
    for i, value in enumerate(sample["drug"]):
        if value == "true":
            sample["drug"][i] = 1
        elif value == "false":
            sample["drug"][i] = 0

for sample in samples:
    for i, value in enumerate(sample["drug"]):
        if value != "nan":
            drug_sums[i] += float(value)
            drug_counts[i] += 1
            drug_avg = drug_sums / np.where(drug_counts == 0, 1, drug_counts)
for sample in samples:
    for i, value in enumerate(sample["drug"]):
        if value == "nan":
            sample["drug"][i] = drug_avg[i]


# 标准化函数
def normalize_samples(samples):
    # 收集所有的 feature 和 drug 数据用于计算全局均值和方差
    all_features = np.array([sample['feature'] for sample in samples], dtype=np.float32)
    all_drugs = np.array([sample['drug'] for sample in samples], dtype=np.float32)

    # 分别标准化 feature 和 drug
    feature_scaler = StandardScaler()
    drug_scaler = StandardScaler()

    all_features_scaled = feature_scaler.fit_transform(all_features)
    all_drugs_scaled = drug_scaler.fit_transform(all_drugs)

    # 更新回 samples
    for i, sample in enumerate(samples):
        sample['feature'] = all_features_scaled[i].tolist()
        sample['drug'] = all_drugs_scaled[i].tolist()

    return feature_scaler, drug_scaler

# 调用标准化函数
feature_scaler, drug_scaler = normalize_samples(samples)

def prepare_data(samples):
    X, y = [], []
    for sample in samples:
        features = sample['feature'] + sample['drug']
        time_points = sample['time']
        releases = sample['release_percentage']
        for i, t in enumerate(time_points):
            X.append(features + [t])  # 将时间点和药物特征拼接
            y.append(releases[i])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X, y

X, y = prepare_data(samples)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建PINN模型
input_features = Input(shape=(21,))
x = layers.Dense(64, activation="relu")(input_features)
x = layers.Dense(32, activation="relu")(x)
x = layers.Dense(16, activation="relu")(x)
output = layers.Dense(1)(x)
model = Model(inputs=input_features, outputs=output)


# 物理损失函数：假设 dC/dt = -k * C
def phys_loss(y_true, y_pred):
    # 假设 y_true 的最后一列是时间
    time_points = y_true[:, -1]
    features = y_true[:, :-1]

    # 使用特征和时间点组合来预测 y_pred_t
    with tf.GradientTape() as tape:
        tape.watch(time_points)
        y_pred_t = model(tf.concat([features, tf.expand_dims(time_points, axis=1)], axis=1))

    dy_dt = tape.gradient(y_pred_t, time_points)
    k = 1e-10  # 假设的常数k值
    physics_error = dy_dt + k * y_pred_t  # 根据扩散模型的假设
    return tf.reduce_mean(tf.square(physics_error))

# 物理损失函数：weibull模型
def phys_loss_weibull(y_true, y_pred):
    # 假设 y_true 的最后一列是时间
    time_points = y_true[:, -1]
    features = y_true[:, :-1]

    # 使用特征和时间点组合来预测 y_pred_t
    with tf.GradientTape() as tape:
        tape.watch(time_points)
        y_pred_t = model(tf.concat([features, tf.expand_dims(time_points, axis=1)], axis=1))

    beta = 0.265  # 先快后慢地释放（plga在0.7到1.3之间）
    alpha = 4.24   # 时间尺度5h
    y_model_t = 1 - tf.exp(-tf.pow(time_points / alpha, beta))
    physics_error = tf.abs(y_pred_t - y_model_t)
    return tf.reduce_mean(tf.square(physics_error))


def diffusion_and_degradation_loss(y_true, y_pred):
    """
    物理损失函数，考虑扩散与降解机制。

    - model: 神经网络模型，输出药物释放百分比。
    - x: 时间点或独立变量（时间、载体、药物特征等）。
    - release_type: 选择使用的释放模型类型（combined表示结合扩散与降解）。
    - k1: 扩散常数。
    - k_deg: 降解常数。
    - M_inf: 最终释放量（通常设为1）。
    """
    k1 = 0.1
    k_deg = 0.05
    M_inf = 1
    # 假设 y_true 的最后一列是时间
    time_points = y_true[:, -1]
    features = y_true[:, :-1]

    # 使用特征和时间点组合来预测 y_pred_t
    with tf.GradientTape() as tape:
        tape.watch(time_points)
        y_pred_t = model(tf.concat([features, tf.expand_dims(time_points, axis=1)], axis=1))

    # 扩散模型 + 降解模型的结合
    diffusion = k1 * tf.sqrt(time_points)  # 扩散部分
    degradation = 1 - tf.exp(-k_deg * time_points)  # 降解部分
    y_theoretical = diffusion + 0.1*degradation  # 结合两部分

    # 计算物理损失：模型预测的释放百分比与理论释放百分比的误差
    return tf.reduce_mean(tf.square(y_pred_t - y_theoretical))  # 均方误差损失

# 定义总损失函数(可选择合适的物理模型)
def custom_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    # p_loss = phys_loss_weibull(tf.concat([X_train[:, :-1], y_train[:, None]], axis=1), y_pred)
    # return mse_loss + 0.1 * p_loss
    return mse_loss




# 编译模型
# model.compile(optimizer='adam', loss=custom_loss, metrics=['mean_squared_error'])

model.compile(optimizer=Adam(learning_rate=0.001), loss=custom_loss, metrics=['mean_squared_error'])


# 早停回调函数
early_stopping = EarlyStopping(
    monitor='val_loss',  # 监控验证集的损失
    patience=30,         # 若连续30个epoch验证集损失没有改善，则停止训练
    restore_best_weights=True  # 恢复验证集损失最优的权重
)
# 训练模型
history = model.fit(X_train, y_train, epochs=500, batch_size=4, validation_data=(X_val, y_val), callbacks=[early_stopping])

# 计算最终的R²值
y_val_pred = model.predict(X_val)
final_r2 = r2_score(y_val, y_val_pred)

# 输出最终的R²
print(f"Final R²: {final_r2}")

# 获取最佳 epoch 的索引和训练损失
best_epoch = np.argmin(history.history['val_loss'])
best_val_loss = history.history['val_loss'][best_epoch]
best_train_loss = history.history['loss'][best_epoch]

print(f"最佳 epoch: {best_epoch + 1}")
print(f"最佳 epoch 的验证损失: {best_val_loss:.4f}")
print(f"最佳 epoch 的训练损失: {best_train_loss:.4f}")


# 分类规则：手动设置并通过映射表定义
def plot_results_with_manual_categories(history, samples, model, category_map):
    # 绘制训练和验证损失
    plt.figure(figsize=(24, 12))

    # 第一张图：loss 曲线
    plt.subplot(2, 4, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(False)

    # 使用分类映射表将样本分组
    categories = sorted(set(category_map.values()))  # 提取所有类别
    for category_index, category in enumerate(categories):
        # 获取属于该类别的样本
        selected_samples = [
            sample for sample, label in zip(samples, category_map.values())
            if label == category
        ]

        # 绘制每个类别的预测曲线
        plt.subplot(2, 4, category_index + 2)
        for i, sample in enumerate(selected_samples):
            # 准备单个样本的数据
            X_sample, y_sample = prepare_data([sample])
            pred_values = model.predict(X_sample).flatten()

            # 绘制真实值和预测值
            plt.plot(
                sample['time'], y_sample, 'o', label=f'True {i + 1}', color=f'C{i}'
            )
            plt.plot(
                sample['time'], pred_values, '-', linewidth=2, label=f'Pred {i + 1}', color=f'C{i}'
            )

        plt.xlabel('Time Points')
        plt.ylabel('Release Percentage')
        plt.title(f'Category {category}')
        plt.grid(False)
        plt.legend()

    plt.tight_layout()
    plt.show()


# 手动将样本编号映射到类别
category_map = {
    1: 'A', 2: 'A', 3: 'A', 4: 'A', 5: 'B',
    6: 'C', 7: 'D', 8: 'E', 9: 'F', 10: 'D',
    11: 'B', 12: 'B', 13: 'F', 14: 'C', 15: 'C',
    16: 'C', 17: 'C', 18: 'B', 19: 'F', 20: 'C',
    21: 'F', 22:'C', 23: 'F', 24: 'D', 25: 'C',
    26: 'F', 27: 'F', 28: 'G', 29: 'A', 30: 'F',
    31: 'F', 32: 'E', 33: 'A', 34: 'B', 35: 'A',
    36: 'F', 37: 'F', 38: 'D', 39: 'B', 40: 'E',
}

# 调用绘图函数
plot_results_with_manual_categories(history, samples, model, category_map)
