import sys
sys.path.append('.')

from warehouses_conf import *
from Offline_Train.train_conf import cluster_K_volatile
from Offline_Train.train_conf import burst_map
from Offline_Train.train_conf import centers_cluster_second_stable
from Monitor.monitor import calculate_warehouses
from Monitor.monitor import sampling

# (capacity, stable_IO, stable_bandwidth, burst_IO, burst_bandwidth)
warehouses = [[0 for _ in range(5)] for _ in range(warehouse_number)]

warehouses_burst_count = [[0 for _ in range(cluster_K_volatile)] for _ in range(warehouse_number)]

warehouses_flag = [1 for _ in range(warehouse_number)]

warehouses_total_time = [[[0 for _ in range(5)] for _ in range(warehouse_number)] for _ in range(evaluate_time_number)]

# [0] (disk_name)
# [1:6] (disk_capacity, disk_if_local, disk_type, vm_cpu, vm_mem)
# [6:10] (ave_disk_IOPS, ave_disk_bandwidth, peak_disk_IOPS, peak_disk_bandwidth)
# [10] (disk_timestamp_num)
# [11] (disk_source_file)
def put_item(item, item_classify_label, item_predictor_label, action):

    item_IO_predict = centers_cluster_second_stable[item_predictor_label][0]
    item_bandwidth_predict = centers_cluster_second_stable[item_predictor_label][1]
    
    if item_classify_label == 0:
        warehouses[action][0] += item[1]
        warehouses[action][1] += item_IO_predict
        warehouses[action][2] += item_bandwidth_predict
    elif item_classify_label == 3:
        warehouses[action][0] += item[1]
        warehouses[action][1] += item_IO_predict
        warehouses[action][2] += item_bandwidth_predict
        warehouses_burst_count[action][burst_map.index(item_predictor_label)] += 1
    
    sampling(item)
    calculate_warehouses(item, action, item_classify_label)
    
    return