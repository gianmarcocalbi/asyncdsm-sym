def compute_value(self, X, y, w):
    return np.sum(1 - y * X.dot(w) / len(y))
