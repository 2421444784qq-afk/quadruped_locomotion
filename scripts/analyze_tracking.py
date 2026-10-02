import numpy as np

data = np.genfromtxt(
    "fixed_base_tracking.csv",
    delimiter=",",
    names=True
)

t = data["time"]
err = data["position_error"]

# 轨迹在仿真时间 3 秒后正式启动
trajectory_time = t - 3.0
phase = np.mod(trajectory_time, 2.0) / 2.0

# 排除前面的站立和过渡阶段
active = trajectory_time >= 0

# 支撑相：周期的前 60%
stance = active & (phase < 0.6)

# 摆动相：周期的后 40%
swing = active & (phase >= 0.6)

def report(name, mask):
    e = err[mask]

    print(f"\n{name}")
    print(f"样本数: {len(e)}")
    print(f"平均误差: {np.mean(e):.6f} m")
    print(f"RMSE: {np.sqrt(np.mean(e**2)):.6f} m")
    print(f"最大误差: {np.max(e):.6f} m")

print("========== 分阶段跟踪误差 ==========")
report("整体周期", active)
report("支撑相", stance)
report("摆动相", swing)

# 分别计算 X、Y、Z 方向误差
for axis in ["x", "y", "z"]:
    e = data[f"{axis}_des"] - data[f"{axis}_actual"]

    print(f"\n{axis.upper()} 方向")
    print(f"平均绝对误差: {np.mean(np.abs(e[active])):.6f} m")
    print(f"最大绝对误差: {np.max(np.abs(e[active])):.6f} m")
