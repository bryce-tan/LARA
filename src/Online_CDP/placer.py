import sys
sys.path.append('.')


from Warehouses.warehouses_conf import warehouse_number
from Warehouses.warehouses_conf import max_bandwidth
from Warehouses.warehouses_conf import max_capacity
from Warehouses.warehouses_conf import max_IOPS
from CDP_conf import reservation_rate
from Offline_Train.train_conf import burst_map
from Offline_Train.train_conf import centers_cluster_second_stable
from Offline_Train.train_conf import multivar_piecewise
from Offline_Train.train_conf import popt_ave_IO
from Offline_Train.train_conf import popt_ave_bandwidth
from Offline_Train.train_conf import popt_peak_IO
from Offline_Train.train_conf import popt_peak_bandwidth

from Monitor.monitor import check_warehouse_burst_count
from Monitor.monitor import check_warehouse_flag
from Monitor.monitor import check_warehouses

# [0] (disk_name)
# [1:6] (disk_capacity, disk_if_local, disk_type, vm_cpu, vm_mem)
# [6:10] (ave_disk_IOPS_fact, ave_disk_bandwidth_fact, peak_disk_IOPS_fact, peak_disk_bandwidth_fatc)
# [10] (disk_timestamp_num)
# [11] (disk_source_file)
def CDA_policy(item, item_label_first, item_label_second):
    action = -1

    # stable disk
    if item_label_first == 0:
        min_manhatten = 4
        for i in range(warehouse_number):
            
            warehouse_flag = check_warehouse_flag(i)
            warehouse_burst_count = check_warehouse_burst_count(i)
            warehouse = check_warehouses(i)

            if warehouse_flag != 1:
                continue

            ave_IO_fit = multivar_piecewise([warehouse_burst_count], *popt_ave_IO)
            ave_bandwidth_fit = multivar_piecewise([warehouse_burst_count], *popt_ave_bandwidth)
            peak_IO_fit = multivar_piecewise([warehouse_burst_count], *popt_peak_IO)
            peak_bandwidth_fit = multivar_piecewise([warehouse_burst_count], *popt_peak_bandwidth)

            item_IO_predict = centers_cluster_second_stable[item_label_second][0]
            item_bandwidth_predict = centers_cluster_second_stable[item_label_second][1]
            
            if warehouse[0]+item[1]<=max_capacity[i]*reservation_rate and warehouse[1]+ave_IO_fit+item_IO_predict<=max_IOPS[i]*reservation_rate and warehouse[2]+ave_bandwidth_fit+item_bandwidth_predict<=max_bandwidth[i]*reservation_rate and peak_IO_fit<=max_IOPS[i] and peak_bandwidth_fit<=max_bandwidth[i]:
                w1 = (warehouse[0] + item[1]) / max_capacity[i]
                w2 = (warehouse[1] + ave_IO_fit + item_IO_predict) / max_IOPS[i]
                w3 = (warehouse[2] + ave_bandwidth_fit + item_bandwidth_predict) / max_bandwidth[i]
                w4 = (w1 + w2 + w3) / 3
                now_manhatten = abs(w1 - w4) + abs(w2 - w4) + abs(w3 - w4)
                if now_manhatten < min_manhatten:
                    action = i
                    min_manhatten = now_manhatten
    
    # volatile disk
    elif item_label_first == 3:
        min_manhatten = 4
        for i in range(warehouse_number):

            warehouse_flag = check_warehouse_flag(i)
            warehouse_burst_count = check_warehouse_burst_count(i)
            warehouse = check_warehouses(i)

            if warehouse_flag != 1:
                continue

            fit_X = warehouse_burst_count[:]
            fit_X[burst_map.index(item_label_second)] += 1
            
            ave_IO_fit = multivar_piecewise([fit_X], *popt_ave_IO)
            ave_bandwidth_fit = multivar_piecewise([fit_X], *popt_ave_bandwidth)
            peak_IO_fit = multivar_piecewise([fit_X], *popt_peak_IO)
            peak_bandwidth_fit = multivar_piecewise([fit_X], *popt_peak_bandwidth)

            if warehouse[0]+item[1]<=max_capacity[i]*reservation_rate and warehouse[1]+ave_IO_fit<=max_IOPS[i]*reservation_rate and warehouse[2]+ave_bandwidth_fit<=max_bandwidth[i]*reservation_rate and peak_IO_fit<=max_IOPS[i]*1 and peak_bandwidth_fit<=max_bandwidth[i]*1:
                w1 = (warehouse[0] + item[1]) / max_capacity[i]
                w2 = (warehouse[1] + ave_IO_fit) / max_IOPS[i]
                w3 = (warehouse[2] + ave_bandwidth_fit) / max_bandwidth[i]
                w4 = (w1 + w2 + w3) / 3
                now_manhatten = abs(w1 - w4) + abs(w2 - w4) + abs(w3 - w4)
                if now_manhatten < min_manhatten:
                    action = i
                    min_manhatten = now_manhatten
    
    return action