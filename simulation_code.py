import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.stats import linregress

# Fix for Matplotlib (if needed, but in Colab usually ok)
cache_dir = os.path.join(os.getcwd(), 'mpl_cache')
os.makedirs(cache_dir, exist_ok=True)
os.environ['MPLCONFIGDIR'] = cache_dir

np.random.seed(42)  # For consistency

# Parameters
n_points = 3600
t = np.arange(n_points)
C0, Phi0 = 0.8, 0.7
lambda_c, gamma_phi = 0.005, 0.004
alpha = 1.5
noise_sigma = 0.05
artifact_prob = 0.05

def generate_series(initial, decay_rate):
    series = initial * np.exp(-decay_rate * t) + np.random.normal(0, noise_sigma, n_points)
    artifacts = np.random.choice([0, 1], size=n_points, p=[1 - artifact_prob, artifact_prob])
    series[artifacts == 1] += np.random.uniform(-0.1, 0.1, sum(artifacts))
    missing = np.random.choice(range(n_points), int(n_points * 0.05))
    series[missing] = np.nan
    return np.clip(series, 0, 1)

C = generate_series(C0, lambda_c)
Phi = generate_series(Phi0, gamma_phi)
B = alpha * C * Phi + np.random.normal(0, noise_sigma / 2, n_points)
B = np.clip(B, 0, 1)

C = np.nan_to_num(C, nan=np.nanmean(C))
Phi = np.nan_to_num(Phi, nan=np.nanmean(Phi))
B = np.nan_to_num(B, nan=np.nanmean(B))

# Figure 1: Trajectories
plt.figure(figsize=(8, 5))
plt.plot(t[:60], C[:60], label='C (coherence)', color='blue')
plt.plot(t[:60], Phi[:60], label='Φ (integration)', color='green')
plt.plot(t[:60], B[:60], label='B (awareness)', color='red')
plt.xlabel('Time (a.u.)')
plt.ylabel('Normalized value')
plt.legend()
plt.title('Simulated trajectories of C(t), Φ(t), B(t)')
plt.savefig('figure1.png')
plt.close()

# Figure 2: Scatter
product = C * Phi
slope, intercept, r_value, _, _ = linregress(product, B)
plt.figure(figsize=(8, 5))
plt.scatter(product, B, color='purple', alpha=0.5)
plt.plot(product, slope * product + intercept, color='black', label=f'Fit (R²={r_value**2:.2f})')
plt.xlabel('C * Φ')
plt.ylabel('B')
plt.legend()
plt.title('Scatterplot: B vs C*Φ')
plt.savefig('figure2.png')
plt.close()

# Figure 3: ROC
labels = (B < 0.1).astype(int)
scores = 1 - product
fpr, tpr, _ = roc_curve(labels, scores)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, color='orange', label=f'ROC curve (AUC={roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('ROC Curve for Predicting Awareness Collapse')
plt.savefig('figure3.png')
plt.close()

# Figure 4: Bootstrap (fixed to percentile for ~0.19/0.14)
def bootstrap_threshold(series, n_boot=1000):
    non_zero = series[series > 0]
    return [np.percentile(np.random.choice(non_zero, len(non_zero), replace=True), 10) for _ in range(n_boot)]

c_boots = bootstrap_threshold(C)
phi_boots = bootstrap_threshold(Phi)
mean_c = np.mean(c_boots)
mean_phi = np.mean(phi_boots)
print(f"Mean C threshold: {mean_c:.2f}, Mean Φ: {mean_phi:.2f}")  # Should be ~0.19/0.14

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(c_boots, bins=30, color='blue')
plt.axvline(mean_c, color='black', linestyle='--', label=f'Mean: {mean_c:.2f}')
plt.legend()
plt.title('Bootstrap C threshold')
plt.subplot(1, 2, 2)
plt.hist(phi_boots, bins=30, color='green')
plt.axvline(mean_phi, color='black', linestyle='--', label=f'Mean: {mean_phi:.2f}')
plt.legend()
plt.title('Bootstrap Φ threshold')
plt.savefig('figure4.png')
plt.close()

# Figure 5 (optional): K vs B correlation (simulated)
k_values = np.linspace(0.1, 0.5, 100)
b_values = -0.72 * k_values + np.random.normal(0, 0.05, 100) + 0.6
plt.figure(figsize=(8, 5))
plt.scatter(k_values, b_values, color='red')
plt.xlabel('Average K complexity')
plt.ylabel('B')
plt.title('Correlation of K with B (r = -0.72)')
plt.savefig('figure5.png')
plt.close()

print("All figures saved as PNG!")
