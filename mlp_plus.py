import datetime
import json
import os

import numpy as np
import pytz
import argparse
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from keras import Input, Model
from keras.api import layers, optimizers
from keras.src.callbacks import EarlyStopping
from keras.src.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Temp', help='model name')
    parser.add_argument('--train_path', type=str, default='./original_data.json',
                        help='path to FSD-MIX-CLIPS_data folder)')
    parser.add_argument('--test_path', type=str, default='./test.json', help='path to FSD-MIX-CLIPS_data folder)')

    parser.add_argument('--train', type=bool, default=False, help='train model')
    return parser.parse_args()


def perprocessing_data(samples):
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

    return samples


def load_train_dataset(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        final_data = json.load(f)
        for i in range(1, 41):
            if i == 37:
                continue
            samples.append(final_data[str(i)])
    samples = perprocessing_data(samples)
    return samples


def load_test_dataset(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        final_data = json.load(f)
        for i in range(33, 41):
            samples.append(final_data[str(i)])
    samples = perprocessing_data(samples)
    return samples


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
    # import pdb; pdb.set_trace()
    # 更新回 samples
    for i, sample in enumerate(samples):
        sample['feature'] = all_features_scaled[i].tolist()
        sample['drug'] = all_drugs_scaled[i].tolist()
    # import pdb; pdb.set_trace()
    return feature_scaler, drug_scaler


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


def build_model():
    # 构建PINN模型
    input_features = Input(shape=(21,))
    x = layers.Dense(64, activation="relu")(input_features)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    output = layers.Dense(1)(x)
    model = Model(inputs=input_features, outputs=output)
    return model


# 定义总损失函数(可选择合适的物理模型)
def custom_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    return mse_loss


def train(model, custom_loss, X_train, y_train, X_val, y_val):
    model.compile(optimizer=Adam(learning_rate=0.001), loss=custom_loss, metrics=['mean_squared_error'])

    # 早停回调函数
    early_stopping = EarlyStopping(
        monitor='val_loss',  # 监控验证集的损失
        patience=30,  # 若连续30个epoch验证集损失没有改善，则停止训练
        restore_best_weights=True  # 恢复验证集损失最优的权重
    )
    # 训练模型
    history = model.fit(X_train, y_train, epochs=500, batch_size=4, validation_data=(X_val, y_val),
                        callbacks=[early_stopping])

    return model, history


def plot_loss(history_loss, history_val_loss):
    # 绘制训练和验证损失
    plt.figure(figsize=(6, 6))
    plt.plot(history_loss, label='Training Loss')
    plt.plot(history_val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('loss.png')


def plot_results_with_manual_categories(history_loss, history_val_loss, samples, model, category_map):
    os.makedirs("pic-category", exist_ok=True)
    # 使用分类映射表将样本分组
    categories = sorted(set(category_map.values()))  # 提取所有类别
    for category_index, category in enumerate(categories):
        # 获取属于该类别的样本
        selected_samples = [
            sample for sample, label in zip(samples, category_map.values())
            if label == category
        ]

        plt.figure(figsize=(6, 6))
        for i, sample in enumerate(selected_samples):
            # 准备单个样本的数据
            X_sample, y_sample = prepare_data([sample])
            pred_values = model.predict(X_sample).flatten()

            # 绘制真实值和预测值
            plt.plot(
                sample['time'], y_sample, 'o', markersize=8, label=f'True {i + 1}', color=f'C{i}'
            )
            plt.plot(
                sample['time'], pred_values, '-', linewidth=4, label=f'Pred {i + 1}', color=f'C{i}'
            )

        plt.xlabel('Time Points (h)')
        plt.ylabel('Release Percentage (%)')
        plt.title(f'Category {category}')
        plt.grid(False)
        # 保存每个类别的图形，文件名为 "category_X.png" 其中 X 是类别编号或名称
        plt.tight_layout()


        save_path = os.path.join("pic-category", f'results_category_{category}.png')
        plt.savefig(save_path)
        plt.close()




def plot_results_with_sample40(samples, model):
    os.makedirs("pic-our-5", exist_ok=True)
    # import pdb; pdb.set_trace()
    selected_samples = [1, 2, 24, 35, 40]
    # selected_samples = [37]
    for i in selected_samples:
        if i == 40:
            i -= 1
        sample = samples[i - 1]
        # sample = samples[-1]
        X_sample, y_sample = prepare_data([sample])
        pred_values = model.predict(X_sample).flatten()
        plt.figure(figsize=(6, 6))
        # 绘制真实值和预测值
        plt.plot(
            sample['time'], y_sample, 'o', markersize=8, label=f'True'
        )
        plt.plot(
            sample['time'], pred_values, '-', linewidth=4, label=f'Pred'
        )

        plt.xlabel('Time Points (h)')
        plt.ylabel('Release Percentage (%)')
        # plt.title(f'Our results for sample {i}')
        plt.title(f'DrugNet Predict {sample["drug_name"]}')
        plt.tight_layout()

        save_path = os.path.join("pic-our-5", f'results_{i}_our.png')
        plt.savefig(save_path)
        plt.close()



def get_MSE_R2(samples, model):
    # import pdb; pdb.set_trace()
    # selected_samples = [1, 2, 24, 35, 37, 40]
    mse_list = []
    r2_list = []

    for sample in samples:
        # sample = samples[i-1]
        # sample = samples[-1]
        X_sample, y_sample = prepare_data([sample])
        pred_values = model.predict(X_sample).flatten()
        mse = mean_squared_error(y_sample, pred_values)
        r2 = r2_score(y_sample, pred_values)
        mse_list.append(mse)
        r2_list.append(r2)
    return mse_list, r2_list


if __name__ == '__main__':
    date_now = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    print(f'Time & Date = {date_now.strftime("%I:%M %p")} , {date_now.strftime("%d_%b_%Y")} \n')
    print(f"*" * 20 + "Begin experiment" + "*" * 20)
    args = get_parse()

    # 设置全局字体为 Times New Roman
    plt.rcParams.update({
        'font.family': 'serif',  # 设置字体为衬线字体
        'font.serif': ['Times New Roman'],  # 设置为 Times New Roman
        'axes.titlesize': 30,  # 设置标题字体大小
        'axes.labelsize': 24,  # 设置坐标轴标签字体大小
        'xtick.labelsize': 20,  # 设置 x 轴刻度标签字体大小
        'ytick.labelsize': 20,  # 设置 y 轴刻度标签字体大小
        'legend.fontsize': 20,  # 设置图例字体大小
    })

    print(f'Time & Date = {date_now.strftime("%I:%M %p")} , {date_now.strftime("%d_%b_%Y")} \n')
    print(f"*" * 20 + "loading dataset" + "*" * 20)
    samples = load_train_dataset(args.train_path)

    # 调用标准化函数
    feature_scaler, drug_scaler = normalize_samples(samples)
    X, y = prepare_data(samples)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

    # import pdb; pdb.set_trace()

    print(f'Time & Date = {date_now.strftime("%I:%M %p")} , {date_now.strftime("%d_%b_%Y")} \n')
    print(f"*" * 20 + "build model" + "*" * 20)
    model = build_model()

    print(f'Time & Date = {date_now.strftime("%I:%M %p")} , {date_now.strftime("%d_%b_%Y")} \n')
    print(f"*" * 20 + "begin train" + "*" * 20)
    if args.train:
        model, history = train(model, custom_loss, X_train, y_train, X_val, y_val)
        # 计算最终的R²值
        y_val_pred = model.predict(X_val)
        final_r2 = r2_score(y_val, y_val_pred)
        # 输出最终的R²
        print(f"Final R²: {final_r2}")

        # 获取最佳 epoch 的索引和训练损失

        history_val_loss = history.history['val_loss']
        history_loss = history.history['loss']
    else:
        model.load_weights("model.weights.h5")
        history_loss = json.load(open("history_loss.json", "r"))
        history_val_loss = json.load(open("history_val_loss.json", "r"))

    best_epoch = np.argmin(history_val_loss)
    best_val_loss = history_val_loss[best_epoch]
    best_train_loss = history_loss[best_epoch]
    print(f"最佳 epoch: {best_epoch + 1}")
    print(f"最佳 epoch 的验证损失: {best_val_loss:.4f}")
    print(f"最佳 epoch 的训练损失: {best_train_loss:.4f}")

    # 手动将样本编号映射到类别
    category_map = {
        1: 'C', 2: 'C', 3: 'C', 4: 'C', 5: 'D',
        6: 'F', 7: 'B', 8: 'G', 9: 'E', 10: 'B',
        11: 'D', 12: 'D', 13: 'E', 14: 'F', 15: 'F',
        16: 'F', 17: 'B', 18: 'D', 19: 'E', 20: 'F',
        21: 'E', 22: 'F', 23: 'E', 24: 'B', 25: 'F',
        26: 'E', 27: 'E', 28: 'A', 29: 'C', 30: 'E',
        31: 'E', 32: 'G', 33: 'C', 34: 'D', 35: 'C',
        36: 'E', 38: 'B', 39: 'D', 40: 'G',
    }

    # 绘制loss曲线
    plot_loss(history_loss, history_val_loss)
    plot_results_with_manual_categories(history_loss, history_val_loss, samples, model, category_map)

    # 绘制特定sample的结果
    plot_results_with_sample40(samples, model)

    # 计算每一个样本RMSE
    # mse_list, r2_list = get_MSE_R2(samples, model)
    # with open('mse_list.csv', 'w') as f:
    #     f.write('mse,r2\n')
    #     for i in range(len(mse_list)):
    #         f.write(str(mse_list[i]) + ',' + str(r2_list[i]) + '\n')
    # f.close()



