import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("jvie_ev_11.csv")
df2 = pd.read_csv("ev_11.csv")
z = df["real"].to_numpy() + 1j * df["imag"].to_numpy()
z2 = df2["real"].to_numpy() + 1j * df2["imag"].to_numpy()

plt.figure(figsize=(6, 6))
plt.scatter(z.real, z.imag, s=20)
plt.scatter(z2.real, z.imag, s=20)

plt.xlabel("Re")
plt.ylabel("Im")
plt.title("Complex plane")
plt.axhline(0)
plt.axvline(0)
plt.gca().set_aspect("equal", adjustable="box")
plt.grid(True)

plt.show()
