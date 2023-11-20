import sys
sys.path.append('.')

from load_data.load import load_data_pretrain
from load_data.load import load_data_pretrain_fitting
from train import train_classifier
from train import train_estimator

def pre_train():

    # [0] (disk_name)
    # [1:6] (disk_capacity, disk_if_local, disk_type, vm_cpu, vm_mem)
    # [6:10] (ave_disk_IOPS, ave_disk_bandwidth, peak_disk_IOPS, peak_disk_bandwidth)
    # [10] (disk_timestamp_num)
    # [11] (disk_source_file)
    items_pretrain = load_data_pretrain()
    train_classifier(items_pretrain)

    # [0] fit_data_X [0:5] (number_of_1st_volatile, ..., number_of_5th_volatile)
    # [1:5] fit_data_ave_IO  fit_data_ave_bandwidth  fit_data_peak_IO  fit_data_peak_bandwidth
    items_pretrain_fitting = load_data_pretrain_fitting()
    train_estimator(items_pretrain_fitting)

    return