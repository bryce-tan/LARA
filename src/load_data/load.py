import sys
sys.path.append('.')

from Monitor.monitor_conf import items_chosen

# [0] (disk_name)
# [1:6] (disk_capacity, disk_if_local, disk_type, vm_cpu, vm_mem)
# [6:10] (ave_disk_IOPS, ave_disk_bandwidth, peak_disk_IOPS, peak_disk_bandwidth)
# [10] (disk_timestamp_num)
# [11] (disk_source_file)
def load_data_pretrain():
    items = []
    # Here are file paths of CDs information to pretrain
    data_file = ''
    with open(data_file, "r") as f:
        for line in f:
            item = line.strip().split(',')
            # (disk_capacity, disk_if_local, disk_type, vm_cpu, vm_mem)
            item[1:6] = map(int, item[1:6])
            # (ave_disk_IOPS, ave_disk_bandwidth, peak_disk_IOPS, peak_disk_bandwidth)
            item[6:10] = map(float, item[6:10])
            # disk_timestamp_num
            item[10] = int(item[10])
            # (item_source_file)
            item[11] = int(item[11])
            items.append([item[0], item[1],item[2],item[3],item[4],item[5], item[6],item[7],item[8],item[9], item[10], item[11]])
            # print(item[0], "has been sampled!")
    return items

# fit_data_X [0:5] (number_of_1st_volatile, ..., number_of_5th_volatile)
# fit_data_ave_IO  fit_data_ave_bandwidth  fit_data_peak_IO  fit_data_peak_bandwidth
def load_data_pretrain_fitting():
    fit_data_X = []
    fit_data_ave_IO = []
    fit_data_ave_bandwidth = []
    fit_data_peak_IO = []
    fit_data_peak_bandwidth = []

    # Here are file paths to record fitting data to pretrain
    fit_data_file_X = ''
    fit_data_file_ave_IO = ''
    fit_data_file_ave_bandwidth = ''
    fit_data_file_peak_IO = ''
    fit_data_file_peak_bandwidth = ''

    with open(fit_data_file_X, "r") as f:
        for line in f:
            item = line.strip().split(',')
            item[0:5] = map(int, item[0:5])
            fit_data_X.append([item[0], item[1], item[2], item[3], item[4]])
    with open(fit_data_file_ave_IO, "r") as f:
        for line in f:
            item = float(line)
            fit_data_ave_IO.append(item)
    with open(fit_data_file_ave_bandwidth, "r") as f:
        for line in f:
            item = float(line)
            fit_data_ave_bandwidth.append(item)
    with open(fit_data_file_peak_IO, "r") as f:
        for line in f:
            item = float(line)
            fit_data_peak_IO.append(item)
    with open(fit_data_file_peak_bandwidth, "r") as f:
        for line in f:
            item = float(line)
            fit_data_peak_bandwidth.append(item)

    return [fit_data_X, fit_data_ave_IO, fit_data_ave_bandwidth, fit_data_peak_IO, fit_data_peak_bandwidth]

# [0] (disk_name)
# [1:6] (disk_capacity, disk_if_local, disk_type, vm_cpu, vm_mem)
# [6:10] (ave_disk_IOPS, ave_disk_bandwidth, peak_disk_IOPS, peak_disk_bandwidth)
# [10] (disk_timestamp_num)
# [11] (disk_source_file)
def load_data_allocation():
    items = []
    # Here are file_paths of CDs information to simulate CDP
    data_file = ''
    with open(data_file, "r") as f:
        for line in f:
            item = line.strip().split(',')
            # (disk_capacity, disk_if_local, disk_type, vm_cpu, vm_mem)
            item[1:6] = map(int, item[1:6])
            # (ave_disk_IOPS, ave_disk_bandwidth, peak_disk_IOPS, peak_disk_bandwidth)
            item[6:10] = map(float, item[6:10])
            # disk_timestamp_num
            item[10] = int(item[10])
            # (item_source_file)
            item[11] = int(item[11])
            items.append([item[0], item[1],item[2],item[3],item[4],item[5], item[6],item[7],item[8],item[9], item[10], item[11]])
            # print(item[0], "has been sampled!")
    return items

def load_data_update_model():
    return items_chosen