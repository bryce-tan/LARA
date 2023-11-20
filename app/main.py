import random
import os
import numpy as np


import sys
sys.path.append('.')

from Offline_Train.pretrain import pre_train
from Offline_Train.train import update_model
from load_data.load import load_data_allocation
from Online_CDP.classifier import classifier
from Online_CDP.predictor import predictor_stable
from Online_CDP.predictor import predictor_volatile
from Online_CDP.placer import CDA_policy
from Monitor.monitor import check_warehouse_flag
from Warehouses.warehouses import put_item

# main
if __name__ == "__main__":

    pre_train()

    # [0] (disk_name)
    # [1:6] (disk_capacity, disk_if_local, disk_type, vm_cpu, vm_mem)
    # [6:10] (ave_disk_IOPS, ave_disk_bandwidth, peak_disk_IOPS, peak_disk_bandwidth)
    # [10] (disk_timestamp_num)
    # [11] (disk_source_file)
    items = load_data_allocation()
    cnt = 0

    stop_time = 0
    max_stop_times = 20000

    while len(items) > 0:

        item = random.choice(items)

        # classifier
        item_classify_label = classifier(item)

        # predictor
        if item_classify_label == 0:
            item_predictor_label = predictor_stable(item)
        elif item_classify_label == 3:
            item_predictor_label = predictor_volatile(item)

        # placer
        action = CDA_policy(item, item_classify_label, item_predictor_label)
        if action == -1:
            stop_time += 1
            if stop_time == max_stop_times:
                break
            continue
        
        # placement decision & monitor
        put_item(item, item_classify_label, item_predictor_label, action)
        items.remove(item)

        warehouses_flag = check_warehouse_flag(-1)
        if warehouses_flag == -1:
            break
        
        # model update
        cnt += 1
        if cnt % 500 == 0:
            update_model()
