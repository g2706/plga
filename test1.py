import numpy as np
import json


with open("original_data.json", "r", encoding="utf-8") as f:
    final_data = json.load(f)

# 将样本数据存储到一个列表中
samples = []

results = []
print("组号 time release")
for i in range(1, 41):
    if i == 37:
        continue
    samples.append(final_data[str(i)])

for i in range(len(samples)):
    print(i, len(samples[i]['time']))