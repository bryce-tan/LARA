from sklearn.tree import DecisionTreeClassifier

import sys
sys.path.append('.')

from Offline_Train.train_conf import model_classify_second_stable
from Offline_Train.train_conf import model_classify_second_volatile

# [1:6] (disk_capacity, disk_if_local, disk_type, vm_cpu, vm_mem)
def predictor_stable(items_predict):

    data_predict_second = sum([[[x[1], x[2], x[3], x[4], x[5]]] for x in items_predict], [])
    items_predict_labels_second = model_classify_second_stable.predict(data_predict_second)

    return items_predict_labels_second

# [1:6] (disk_capacity, disk_if_local, disk_type, vm_cpu, vm_mem)
def predictor_volatile(items_predict):

    data_predict_second = sum([[[x[1], x[2], x[3], x[4], x[5]]] for x in items_predict], [])
    items_predict_labels_second = model_classify_second_volatile.predict(data_predict_second)

    return items_predict_labels_second