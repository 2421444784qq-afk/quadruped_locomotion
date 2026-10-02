import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt(
    "fixed_base_tracking.csv",
    delimiter=",",
    names=True
)

t = data["time"]

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

labels = ["X (m)", "Y (m)", "Z (m)"]
desired = ["x_des", "y_des", "z_des"]
actual = ["x_actual", "y_actual", "z_actual"]

for i in range(3):
    axes[i].plot(t, data[desired[i]], label="Desired")
    axes[i].plot(t, data[actual[i]], "--", label="Actual")
    axes[i].set_ylabel(labels[i])
    axes[i].grid(True)
    axes[i].legend()

axes[-1].set_xlabel("Time (s)")
fig.suptitle("Fixed-base FR Foot Tracking")
fig.tight_layout()
fig.savefig("fixed_base_tracking.png", dpi=200)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(t, data["position_error"])
plt.xlabel("Time (s)")
plt.ylabel("Position error (m)")
plt.title("FR Foot Position Tracking Error")
plt.grid(True)
plt.tight_layout()
plt.savefig("fixed_base_error.png", dpi=200)
plt.show()

print("最大足端误差:", np.max(data["position_error"]))
print("平均足端误差:", np.mean(data["position_error"]))
