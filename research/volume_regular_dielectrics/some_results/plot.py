from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("/home/evgen/Education/MasterDegree/thesis/my_papers/Utils_for_papers/graph_style.mplstyle")

# Папка с csv-файлами
data_dir = Path(".")

# Ищем нужные файлы
files = sorted(data_dir.glob("sigma_vv_*.csv"))

plt.figure(figsize=(10, 6))

labels = ["12, -5", "20, -3", "20, -2", "20, -5", "теория", "40, -3"]

for file, label in zip(files, labels):
    df = pd.read_csv(file)

    if "angle" not in df.columns or "rsp" not in df.columns:
        raise ValueError(f"В файле {file.name} нет столбцов 'angle' и/или 'rsp'")

    plt.plot(df["angle"], df["rsp"], label=label)

plt.xlabel("Угол наблюдения, град")
plt.ylabel("ЭПР, дБ")
plt.grid(True)
plt.legend(title=f"Количество ячеек на сторону,\n порядок точность аппрокс.")
plt.tight_layout()
plt.show()
