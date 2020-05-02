import numpy as np


X = np.random.uniform(0, 100, (10, 5200))


X_centered = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(X_centered)
W2 = Vt.T[:, :2]
X2D = X_centered.dot(W2)

