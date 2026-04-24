import numpy as np
from scipy.optimize import minimize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# -------------------------------
# 1. Better Data (IMPORTANT FIX)
# -------------------------------
X = np.array([
    [1, 2],
    [2, 3],
    [10, 10],
    [-5, -3],
    [0, 8]
])

# 1 = similar, 0 = not similar
y = np.array([
    [1,1,0,0,0],
    [1,1,0,0,0],
    [0,0,1,0,0],
    [0,0,0,1,0],
    [0,0,0,0,1]
])

# -------------------------------
# 2. Strong Loss Function
# -------------------------------
def loss(weights):
    W = np.diag(weights)

    # Normalize after weighting
    X_t = normalize(X @ W)

    sim = cosine_similarity(X_t)

    # Similar pairs → want 1
    similar_loss = (1 - sim)[y == 1]

    # Dissimilar → want low similarity (margin based)
    dissimilar_loss = np.maximum(0, sim[y == 0] - 0.05)
    

    # Combine (dissimilar ko zyada importance)
    loss_val = np.mean(similar_loss**2) + 4*np.mean(dissimilar_loss**2)


    # Regularization
    reg = 0.1 * np.sum((weights - 1) ** 2)

    return loss_val + reg

# -------------------------------
# 3. Optimization
# -------------------------------
init_w = np.ones(X.shape[1])

result = minimize(
    loss,
    init_w,
    method='L-BFGS-B',
    bounds=[(0.1, 10)] * len(init_w)
)

optimal_weights = result.x

# -------------------------------
# 4. Final Similarity
# -------------------------------
W_opt = np.diag(optimal_weights)
X_opt = normalize(X @ W_opt)
final_sim = cosine_similarity(X_opt)


# -------------------------------
# 5. Output
# -------------------------------
print("✅ Optimal Weights:", optimal_weights)
print("\n📊 Final Similarity Matrix:\n", np.round(final_sim, 3))