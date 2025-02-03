import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

# シミュレーション関数
def bowker_test_statistic(table):
    stat = 0.0
    k = table.shape[0]
    for i in range(k):
        for j in range(i+1, k):
            diff = table[i, j] - table[j, i]
            denom = table[i, j] + table[j, i]
            if denom > 0:
                stat += diff**2 / denom
    return stat

def simulate_bowker_test(n, prob_matrix, alpha=0.05, nsim=5000):
    k = prob_matrix.shape[0]
    df = int(k * (k - 1) / 2)
    crit_val = chi2.ppf(1 - alpha, df)
    rejections = 0

    for _ in range(nsim):
        outcomes = np.random.choice(np.arange(k * k), size=n, p=prob_matrix.flatten())
        table = np.zeros((k, k), dtype=int)
        for outcome in outcomes:
            i, j = divmod(outcome, k)
            table[i, j] += 1

        stat = bowker_test_statistic(table)
        if stat > crit_val:
            rejections += 1

    return rejections / nsim  # 検出力（power）

# Streamlit UI
st.title("Bowker Test Power Simulation")

# ユーザー入力
Linear = st.number_input("Number of Linear samples", min_value=1, value=21)
Spiral = st.number_input("Number of Spiral samples", min_value=1, value=6)
No = st.number_input("Number of No samples", min_value=1, value=2)

# Before の確率行列
before_counts = np.array([Linear, Spiral, No])
transition_probs = np.ones((3,3)) / 3
prob_matrix = transition_probs * before_counts[:, np.newaxis]
prob_matrix /= prob_matrix.sum()

target_power_80 = 0.8
target_power_90 = 0.9
current_power = 0.0
n = 20
step = 5
max_samples = 200

required_sample_size_80 = None
required_sample_size_90 = None

sample_sizes = []
power_estimates = []

st.write("Running simulation...")

while current_power < target_power_90 and n <= max_samples:
    current_power = simulate_bowker_test(n, prob_matrix, alpha=0.05, nsim=5000)
    sample_sizes.append(n)
    power_estimates.append(current_power)

    if required_sample_size_80 is None and current_power >= target_power_80:
        required_sample_size_80 = n

    if required_sample_size_90 is None and current_power >= target_power_90:
        required_sample_size_90 = n

    if required_sample_size_90 is not None:
        break

    n += step

current_total_samples = before_counts.sum()
samples_needed_80 = max(0, required_sample_size_80 - current_total_samples)
samples_needed_90 = max(0, required_sample_size_90 - current_total_samples)

st.write(f"Current total samples: {current_total_samples}")
st.write(f"Required total samples for 80% power: {required_sample_size_80}")
st.write(f"Required total samples for 90% power: {required_sample_size_90}")
st.write(f"Additional samples needed for 80% power: {samples_needed_80}")
st.write(f"Additional samples needed for 90% power: {samples_needed_90}")

# グラフ描画
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sample_sizes, power_estimates, marker='o', linestyle='-', label="Power Curve")
ax.scatter([required_sample_size_80, required_sample_size_90], [0.8, 0.9], color='red', label="Required Sample Sizes")
ax.axhline(y=0.8, color='b', linestyle='--', label="Target Power 80%")
ax.axhline(y=0.9, color='r', linestyle='--', label="Target Power 90%")
ax.set_xlabel("Sample size (n)")
ax.set_ylabel("Estimated Power")
ax.set_title("Bowker Test Power Simulation")
ax.legend()
ax.grid(True)
st.pyplot(fig)
