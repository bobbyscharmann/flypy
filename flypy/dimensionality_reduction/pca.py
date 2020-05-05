import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
X, _ = make_swiss_roll() #np.random.uniform(0, 100, (10, 5200))


X_centered = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(X_centered)
W2 = Vt.T[:, :2]
X2D = X_centered.dot(W2)
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(X[:, 0], X[:, 1], X[:, 2])
plt.scatter(X2D[:, 0], X2D[:, 1])
plt.show()