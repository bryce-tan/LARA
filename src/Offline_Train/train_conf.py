import numpy as np
from sklearn.tree import DecisionTreeClassifier

model_classify_first = DecisionTreeClassifier(criterion="gini")
model_classify_second_stable = DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_split=10)
model_classify_second_volatile = DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_split=10)
popt_ave_IO = []
popt_ave_bandwidth = []
popt_peak_IO = []
popt_peak_bandwidth = []

burst_map = []
centers_cluster_second_stable = []

# classifier
IO_line = 100
bandwidth_line = 25

# predictor
cluster_K_stable = 5
cluster_K_volatile = 5

# estimator
def multivar_piecewise(x_new, a1, b11, b12, b13, b14, b15, a2, b21, b22, b23, b24, b25, a3, b31, b32, b33, b34, b35, a4, b41, b42, b43, b44, b45, a5, b51, b52, b53, b54, b55):
    # 
    x = np.array(x_new)
    y = np.zeros(x.shape[0])
    #
    mask1 = x[:, 4] <= 200
    y[mask1] = a1 + b11 * x[mask1, 0] + b12 * x[mask1, 1] + b13 * x[mask1, 2] + b14 * x[mask1, 3] + b15 * x[mask1, 4]
    #
    mask2 = (x[:, 4] > 200) & (x[:, 4] <= 400)
    y[mask2] = a2 + b21 * x[mask2, 0] + b22 * x[mask2, 1] + b23 * x[mask2, 2] + b24 * x[mask2, 3] + b25 * x[mask2, 4]
    # 
    mask3 = (x[:, 4] > 400) & (x[:, 4] <= 600)
    y[mask3] = a3 + b31 * x[mask3, 0] + b32 * x[mask3, 1] + b33 * x[mask3, 2] + b34 * x[mask3, 3] + b35 * x[mask3, 4]
    # 
    mask4 = (x[:, 4] > 600) & (x[:, 4] <= 800)
    y[mask4] = a4 + b41 * x[mask4, 0] + b42 * x[mask4, 1] + b43 * x[mask4, 2] + b44 * x[mask4, 3] + b45 * x[mask4, 4]
    # 
    mask5 = x[:, 4] > 800
    y[mask5] = a5 + b51 * x[mask5, 0] + b52 * x[mask5, 1] + b53 * x[mask5, 2] + b54 * x[mask5, 3] + b55 * x[mask5, 4]
    return y
