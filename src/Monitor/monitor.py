import os
import linecache

import sys
sys.path.append('.')

from Warehouses.warehouses import warehouses
from Warehouses.warehouses import warehouses_burst_count
from Warehouses.warehouses import warehouses_flag
from Warehouses.warehouses import warehouses_total_time
from Warehouses.warehouses_conf import warehouse_number
from Warehouses.warehouses_conf import max_IOPS
from Warehouses.warehouses_conf import max_bandwidth
from monitor_conf import *

def check_warehouse_flag(choice):
    if choice == -1:
        for i in range(warehouse_number):
            if warehouses_flag[i] != -1:
                return 1
        return -1
    else:
        return warehouses_flag[choice]

def check_warehouse_burst_count(choice):
    return warehouses_burst_count[choice]

def check_warehouses(choice):
    return warehouses[choice]

def sampling(item):
    items_chosen.append(item)
    return

# [0] (disk_name)
# [1:6] (disk_capacity, disk_if_local, disk_type, vm_cpu, vm_mem)
# [6:10] (ave_disk_IOPS, ave_disk_bandwidth, peak_disk_IOPS, peak_disk_bandwidth)
# [10] (disk_timestamp_num)
# [11] (disk_source_file)
def calculate_warehouses(item, action, disk_label_first):
    warehouse_violation_time = 0

    item_dir = ''

    disk_name = item[0]
    disk_capacity = item[1]
    item_source_file = item[11]

    item_dir = item_dir + str(item_source_file) + '/'
    f = os.path.join(item_dir, str(disk_name))

    for i in range(evaluate_time_number):
        
        # [1:5] (IO_count_r, bandwidth_count_r, IO_count_w, bandwidth_count_w) 
        disk_IO_bandwidth = linecache.getline(f, i + 1)
        fields = disk_IO_bandwidth.strip().split(',')
        disk_IO = int(fields[1]) + int(fields[3])
        disk_bandwidth = int(fields[2]) + int(fields[4])

        warehouses_total_time[i][action][0] += disk_capacity
        if disk_label_first == 0:
            warehouses_total_time[i][action][1] += disk_IO
            warehouses_total_time[i][action][2] += disk_bandwidth
        elif disk_label_first == 3:
            warehouses_total_time[i][action][3] += disk_IO
            warehouses_total_time[i][action][4] += disk_bandwidth

        if warehouses_total_time[i][action][1] + warehouses_total_time[i][action][3] > max_IOPS[action] or warehouses_total_time[i][action][2] + warehouses_total_time[i][action][4]> max_bandwidth[action]:
            warehouse_violation_time += 1
    
    if warehouse_violation_time > max_violation_time_line:
        warehouses_flag[action] = -1

    return 0