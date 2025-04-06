import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit
import json


# 3个不同的模型
def model1(t, a, b):
    # weibull模型
    return 100 * (1 - np.exp(-np.power(t / a, b)))


def model2(t, k, n):
    # k-p模型
    return 100 * (k * np.power(t, n))


def model3(x, a, b):
    # 待定，随便写的一个模型，不用理它
    return a * x ** b


# 加载全部样本
with open("./original_data.json", "r", encoding="utf-8") as f:
    final_data = json.load(f)

# 将 40个 样本数据存储到一个列表中
samples = []
for i in range(1, 41):
    if i == 37:
        continue
    samples.append(final_data[str(i)])

# 结果存储
results = []

# 创建保存图片的文件夹
output_dir = "pic-weibull"
os.makedirs(output_dir, exist_ok=True)

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


# 对于每个样本，拟合不同模型并计算 MSE 和 R²
for sample_id, sample in enumerate(samples):
    time_points = np.array(sample['time'])  # 时间点
    release_amounts = np.array(sample['release_percentage'])  # 药物释放量

    # 对于每个模型，拟合并计算 MSE 和 R²
    for model_id, model in enumerate([model1], start=1):
        # 使用 curve_fit 函数拟合模型
        bounds = ([1e-5, 1e-5], [np.inf, np.inf])
        popt, _ = curve_fit(model, time_points, release_amounts, p0=[1, 0.1], bounds=bounds, maxfev=10000)

        # 得到拟合后的参数
        fitted_values = model(time_points, *popt)

        # 计算 MSE 和 R²
        mse = mean_squared_error(release_amounts, fitted_values)
        r2 = r2_score(release_amounts, fitted_values)

        # 将结果保存
        results.append({
            'Sample ID': sample_id ,
            'Model': f'Model 1',
            'MSE': mse,
            'R2': r2,
            'Params': popt.tolist()  # 保存拟合的参数
        })

        # 绘制图像
        t_fine = np.linspace(min(time_points), max(time_points), 100)
        r_fit = model(t_fine, *popt)

        plt.figure(figsize=(6, 6))
        # plt.plot(time_points, release_amounts, 'o', label="Real Data", markersize=6, color='blue')
        # plt.plot(t_fine, r_fit, '--', label="Fitted Curve", color='red')
        # plt.xlabel("Time")
        # plt.ylabel("Release Percentage")
        # plt.title(f"Sample {sample_id + 1}: Weibull Model Fit")
        #
        plt.plot(
            time_points, release_amounts, 'o', markersize=8, label=f'True'
        )
        plt.plot(
            t_fine, r_fit, '-', linewidth=4, label=f'Pred'
        )
        plt.xlabel('Time Points (h)')
        plt.ylabel('Release Percentage (%)')
        plt.title('Weibull Results for Sample {}'.format(sample_id + 1))
        plt.grid(False)
        plt.tight_layout()

        # 保存图像
        output_path = os.path.join(output_dir, 'results_{}_weibull.png'.format(sample_id + 1))
        plt.savefig(output_path)
        print(f"Sample {sample_id + 1} 的图像已保存至：{output_path}")
        plt.close()

# 将结果保存为 Pandas DataFrame
df_results = pd.DataFrame(results)

# 保存为 CSV 文件
df_results.to_csv("weibull_model_results.csv", index=False)

print("拟合结果已保存到 weibull_model_results.csv")
