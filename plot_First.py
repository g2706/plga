import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit
import json

from plga import prepare_data


# 一级动力学模型
def model1(t, k):
    # First Order模型
    return 100 * (1 - np.exp(-k * t))


def plot_results_with_sample40(samples, model):
    os.makedirs("pic-first-5", exist_ok=True)
    # import pdb; pdb.set_trace()
    # sample = samples[-1]
    selected_samples = [1, 2, 24, 35, 40]
    for i in selected_samples:
        if i == 40:
            i -= 1
        sample = samples[i - 1]
        # X_sample, y_sample = prepare_data([sample])
        # 准备单个样本的数据
        plt.figure(figsize=(6, 6))
        time_points = np.array(sample['time'])  # 时间点
        release_amounts = np.array(sample['release_percentage'])  # 药物释放量

        # 使用 curve_fit 函数拟合模型
        bounds = ([-np.inf], [np.inf])
        popt, _ = curve_fit(model, time_points, release_amounts, p0=[0.01], bounds=bounds, maxfev=10000)

        # 得到拟合后的参数
        fitted_values = model(time_points, *popt)

        # 绘制图像
        t_fine = np.linspace(min(time_points), max(time_points), 100)
        r_fit = model(t_fine, *popt)

        # 绘制真实值和预测值
        plt.plot(
            time_points, release_amounts, 'o', markersize=8, label=f'True'
        )
        plt.plot(
            t_fine, r_fit, '-', linewidth=4, label=f'Pred'
        )
        plt.xlabel('Time Points (h)')
        plt.ylabel('Release Percentage (%)')
        plt.title(f'F-O Predict {sample["drug_name"]}')
        plt.tight_layout()

        save_path = os.path.join("pic-first-5", f'results_{i}_first.png')
        plt.savefig(save_path)
        plt.close()

# def results_csv_with_all_samples(samples, model):
#     results = []
#     os.makedirs("results_csv", exist_ok=True)
#     # 对于每个样本，拟合不同模型并计算 MSE 和 R²
#     for sample_id, sample in enumerate(samples):
#         time_points = np.array(sample['time'])  # 时间点
#         release_amounts = np.array(sample['release_percentage'])  # 药物释放量
#
#         # 对于每个模型，拟合并计算 MSE 和 R²
#         for model_id, model in enumerate([model], start=1):
#             # 使用 curve_fit 函数拟合模型
#             bounds = ([-np.inf], [np.inf])
#             popt, _ = curve_fit(model, time_points, release_amounts, p0=[0.01], bounds=bounds, maxfev=10000)
#
#             # 得到拟合后的参数
#             fitted_values = model(time_points, *popt)
#
#             # 计算 MSE 和 R²
#             mse = mean_squared_error(release_amounts, fitted_values)
#             r2 = r2_score(release_amounts, fitted_values)
#
#             # 将结果保存
#             results.append({
#                 'Sample ID': sample_id,
#                 'Model': f'Model 3',
#                 'MSE': mse,
#                 'R2': r2,
#                 'Params': popt.tolist()  # 保存拟合的参数
#             })
#
#     # 将结果保存为 Pandas DataFrame
#     df_results = pd.DataFrame(results)
#     save_path = os.path.join("results_csv", "first_model_results.csv")
#     # 保存为 CSV 文件
#     df_results.to_csv(save_path, index=False)
#
#     print("拟合结果已保存到 first_model_results.csv")

def results_csv_with_all_samples(samples, model, category_map):
    results = []
    category_results = {key: {'mse': [], 'r2': []} for key in category_map.values()}
    all_mse = []
    all_r2 = []

    os.makedirs("results_csv", exist_ok=True)

    # 对于每个样本，拟合不同模型并计算 MSE 和 R²
    for sample_id, sample in enumerate(samples):
        time_points = np.array(sample['time'])  # 时间点
        release_amounts = np.array(sample['release_percentage'])  # 药物释放量

        # 对于每个模型，拟合并计算 MSE 和 R²
        for model_id, model in enumerate([model], start=1):
            # 使用 curve_fit 函数拟合模型
            bounds = ([-np.inf], [np.inf])
            popt, _ = curve_fit(model, time_points, release_amounts, p0=[0.01], bounds=bounds, maxfev=10000)

            # 得到拟合后的参数
            fitted_values = model(time_points, *popt)

            # 计算 MSE 和 R²
            mse = mean_squared_error(release_amounts, fitted_values)
            r2 = r2_score(release_amounts, fitted_values)

            # 将结果保存
            results.append({
                'Sample ID': sample_id,
                'Model': f'Model 3',
                'MSE': mse,
                'R2': r2,
                'Params': popt.tolist(),  # 保存拟合的参数
                'Category': category_map.get(sample_id + 1, 'Unknown')  # 获取样本类别
            })

            # 保存 MSE 和 R² 到对应类别
            category = category_map.get(sample_id + 1, 'Unknown')
            category_results[category]['mse'].append(mse)
            category_results[category]['r2'].append(r2)

            # 保存到全局 MSE 和 R² 列表
            all_mse.append(mse)
            all_r2.append(r2)

    # 计算每个类别的平均 MSE 和 R²
    category_avg_results = {}
    for category, values in category_results.items():
        avg_mse = np.mean(values['mse']) if values['mse'] else None
        avg_r2 = np.mean(values['r2']) if values['r2'] else None
        category_avg_results[category] = {'Average MSE': avg_mse, 'Average R²': avg_r2}

    # 计算全数据集的平均 MSE 和 R²
    avg_mse_all = np.mean(all_mse)
    avg_r2_all = np.mean(all_r2)

    # 打印每个类别的平均 MSE 和 R²
    print("类别平均结果：")
    for category, avg_result in category_avg_results.items():
        print(f"{category} - 平均 MSE: {avg_result['Average MSE']}, 平均 R²: {avg_result['Average R²']}")

    # 打印全数据集的平均 MSE 和 R²
    print(f"\n整个数据集 - 平均 MSE: {avg_mse_all}, 平均 R²: {avg_r2_all}")

    # 将结果保存为 Pandas DataFrame
    df_results = pd.DataFrame(results)
    save_path = os.path.join("results_csv", "first_model_results.csv")
    df_results.to_csv(save_path, index=False)
    print("拟合结果已保存到 first_model_results.csv")


if __name__ == '__main__':
    # 加载全部样本
    with open("./original_data.json", "r", encoding="utf-8") as f:
        final_data = json.load(f)

    # 将 40个 样本数据存储到一个列表中
    samples = []
    for i in range(1, 41):
        if i == 37:
            continue
        samples.append(final_data[str(i)])

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

    category_map = {
        1: 'C', 2: 'C', 3: 'C', 4: 'C', 5: 'D',
        6: 'F', 7: 'B', 8: 'G', 9: 'E', 10: 'B',
        11: 'D', 12: 'D', 13: 'E', 14: 'F', 15: 'F',
        16: 'F', 17: 'B', 18: 'D', 19: 'E', 20: 'F',
        21: 'E', 22: 'F', 23: 'E', 24: 'B', 25: 'F',
        26: 'E', 27: 'E', 28: 'A', 29: 'C', 30: 'E',
        31: 'E', 32: 'G', 33: 'C', 34: 'D', 35: 'C',
        36: 'E', 37: 'E', 38: 'B', 39: 'D', 40: 'G',
    }

    model = model1
    plot_results_with_sample40(samples, model)
    results_csv_with_all_samples(samples, model, category_map)
    # plot_results_with_allsamples(samples, model)
