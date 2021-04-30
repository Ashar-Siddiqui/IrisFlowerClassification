import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

dataset = load_iris()
print(dataset.DESCR)

X = dataset.data
y = dataset.target

plt.plot(X[:, 0][y == 0]* X[:, 1][y == 0], X[:, 1][y == 0]* X[:, 2][y == 0], 'r.', label = 'Setosa')
plt.plot(X[:, 0][y == 1]* X[:, 1][y == 1], X[:, 1][y == 1]* X[:, 2][y == 1], 'g.', label = 'Versicolour')
plt.plot(X[:, 0][y == 2]* X[:, 1][y == 2], X[:, 1][y == 2]* X[:, 2][y == 2], 'b.', label = 'Viriginica')
plt.legend()
plt.show()

from sklearn.preprocessing import StandardScaler
StandardScaler().fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)
log_reg.score(X_test, y_test)
log_reg.score(X, y)
