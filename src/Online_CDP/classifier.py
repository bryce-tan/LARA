from sklearn.tree import DecisionTreeClassifier

import sys
sys.path.append('.')

from Offline_Train.train_conf import model_classify_first

# [1:6] (disk_capacity, disk_if_local, disk_type, vm_cpu, vm_mem)
def classifier(items_predict):

    data_predict_first = sum([[[row[1], row[2], row[3], row[4], row[5]]] for row in items_predict], [])
    items_predict_labels_first = model_classify_first.predict(data_predict_first)
    
    return items_predict_labels_first