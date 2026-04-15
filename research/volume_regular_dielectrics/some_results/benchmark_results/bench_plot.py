from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("/home/evgen/Education/MasterDegree/thesis/my_papers/Utils_for_papers/graph_style.mplstyle")

data1 = [21, 49.8, 105, 199, 379, 622, 999, 1577]
data2 = [118, 84.1, 483, 791, 1290, 1960, 772, 4212]
data3 = [61.9, 123, 233, 422, 679, 1059, 1615, 2461]
N = [20, 24, 28, 32, 36, 40, 44, 48]
N2 = [20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]

data4 = [40, 76.2, 146, 252, 439, 735, 1195, 1879, 2526, 3752, 5526]
data5 = [96.1, 184, 346, 609, 994, 1563, 2389, 3680, 5023, 7200, 10262]
data6 = [118, 84.1, 483, 791, 1290, 1960, 772, 4212, 1674, 7447, 9864]

plt.plot(N2, data4, label="Toeplitz, -3")
plt.plot(N2, data5, label="Toeplitz, -5")
plt.plot(N2, data6, label="FFT")

plt.xlabel("Количество ячеек на сторону куба")
plt.ylabel("Время работы, мс")
plt.yscale('log')
plt.grid(True)
plt.legend(title=f"Стратегия matvec")
plt.tight_layout()
plt.show()
