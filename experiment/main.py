import os
import linecache
import numpy as np
import pandas as pd
import random
from sklearn.cluster import KMeans
from collections import deque
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.model_selection import cross_val_score
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

from scipy.optimize import curve_fit


# simulate_episode
episodes = 1
# max_load_item_number = len(items) - episode_remain
episode_remain = 500
# evaluate_time_number
evaluate_time_number = 10000
# 
time_interval = 1
# 
min_timestamp_num = 10000

# 
max_violation_time_line = 100

# 
warehouse_number = 9

max_capacity = [17000, 17000, 17000, 17000, 17000, 17000, 5000, 5000, 5000]
max_IOPS = [40000, 40000, 40000, 25000, 25000, 25000, 50000, 50000, 50000] 
max_bandwidth = [20000, 20000, 20000, 60000, 60000, 60000, 60000, 60000, 60000]

for i in range(warehouse_number):
    max_capacity[i] *= 10
    max_IOPS[i] *= 10
    max_bandwidth[i] *= 10

warehouse_max = np.array([max_capacity, max_IOPS, max_bandwidth])

# reservation = 1 - reservation_rate
reservation_rate = 0.8

# classify
cluster_first_dimension = 4
IO_line = 100
bandwidth_line = 50

# predictor
cluster_K_second = [5, 5, 5, 5]
IO_predict_center = []
bandwidth_predict_center = []

# max_stop_times
stop_times = 20000

# (time, fact_max)
burst_disk_IO_evaluate = [[0 for _ in range(2)] for _ in range(warehouse_number)]
burst_disk_bandwidth_evaluate = [[0 for _ in range(2)] for _ in range(warehouse_number)]
stable_disk_IO_evaluate = [[0 for _ in range(2)] for _ in range(warehouse_number)]
stable_disk_bandwidth_evaluate = [[0 for _ in range(2)] for _ in range(warehouse_number)]
burst_disk_IO_predict_peak = [0 for _ in range(warehouse_number)]
burst_disk_bandwidth_predict_peak = [0 for _ in range(warehouse_number)]
burst_disk_IO_predict_ave = [0 for _ in range(warehouse_number)]
burst_disk_bandwidth_predict_ave = [0 for _ in range(warehouse_number)]
stable_disk_IO_predict = [0 for _ in range(warehouse_number)]
stable_disk_bandwidth_predict = [0 for _ in range(warehouse_number)]

# train_data
def load_data_train():

    items = []
    # Here are file paths of CD informations
    data_temp = []
    for j in range(len(data_temp)):
        with open(data_temp[j], "r") as f:
            for line in f:
                item = line.strip().split(',')
                # (disk_capacity, disk_if_local, disk_type, vm_cpu, vm_mem)
                item[1:9] = map(int, item[1:9])
                # (ave_disk_IOPS, ave_disk_bandwidth, peak_disk_IOPS, peak_disk_bandwidth)
                item[9:13] = map(float, item[9:13])
                # (disk_timestamp_num)
                item[13] = int(item[13])
                # remove invalid disks
                if item[13] < min_timestamp_num:
                    continue
                # (item_source_file)
                item_source_file = int(j+6)
                items.append([item[0], item[1],item[2],item[4],item[7],item[8], item[9],item[10],item[11],item[12], item[13], item_source_file])
                # print(item[0], "has been sampled!")
    return items

# load_data to simulate
def load_data_predict():

    # Here are file paths of CD informations
    items = []
    data_temp = []
    for j in range(len(data_temp)):
        with open(data_temp[j], "r") as f:
            for line in f:
                item = line.strip().split(',')
                # (disk_capacity, disk_if_local, disk_type, vm_cpu, vm_mem)
                item[1:9] = map(int, item[1:9])
                # (ave_disk_IOPS, ave_disk_bandwidth, peak_disk_IOPS, peak_disk_bandwidth)
                item[9:13] = map(float, item[9:13])
                # (disk_timestamp_num)
                item[13] = int(item[13])
                # remove invalid disks
                if item[13] < min_timestamp_num:
                    continue
                # (item_source_file)
                if j == 0:
                    item_source_file = 0
                elif j == 1:
                    item_source_file = 2
                items.append([item[0], item[1],item[2],item[4],item[7],item[8], item[9],item[10],item[11],item[12], item[13], item_source_file])
                # print(item[0], "has been sampled!")
    return items

def load_data_fit(flag):
    fit_data = []
    fit_data_file = []
    if flag == 0:
        with open(fit_data_file[flag], "r") as f:
            for line in f:
                item = line.strip().split(',')
                item[0:5] = map(int, item[0:5])
                fit_data.append([item[0], item[1], item[2], item[3], item[4]])
    else:
        with open(fit_data_file[flag], "r") as f:
            for line in f:
                item = float(line)
                fit_data.append(item)
    return fit_data

# fitting
def multivar_piecewise(x_new, a1, b11, b12, b13, b14, b15, a2, b21, b22, b23, b24, b25, a3, b31, b32, b33, b34, b35, a4, b41, b42, b43, b44, b45, a5, b51, b52, b53, b54, b55):
    #
    x = np.array(x_new)
    y = np.zeros(x.shape[0])
    # x5<=200
    mask1 = x[:, 4] <= 200
    y[mask1] = a1 + b11 * x[mask1, 0] + b12 * x[mask1, 1] + b13 * x[mask1, 2] + b14 * x[mask1, 3] + b15 * x[mask1, 4]
    # 200<x5<=400
    mask2 = (x[:, 4] > 200) & (x[:, 4] <= 400)
    y[mask2] = a2 + b21 * x[mask2, 0] + b22 * x[mask2, 1] + b23 * x[mask2, 2] + b24 * x[mask2, 3] + b25 * x[mask2, 4]
    # 400<x5<=600
    mask3 = (x[:, 4] > 400) & (x[:, 4] <= 600)
    y[mask3] = a3 + b31 * x[mask3, 0] + b32 * x[mask3, 1] + b33 * x[mask3, 2] + b34 * x[mask3, 3] + b35 * x[mask3, 4]
    # 600<x5<=800
    mask4 = (x[:, 4] > 600) & (x[:, 4] <= 800)
    y[mask4] = a4 + b41 * x[mask4, 0] + b42 * x[mask4, 1] + b43 * x[mask4, 2] + b44 * x[mask4, 3] + b45 * x[mask4, 4]
    # x5>800
    mask5 = x[:, 4] > 800
    y[mask5] = a5 + b51 * x[mask5, 0] + b52 * x[mask5, 1] + b53 * x[mask5, 2] + b54 * x[mask5, 3] + b55 * x[mask5, 4]
    return y

# stable disk or volatile disk
# stable disk - 0 - IO_stable and bandwidth_stable
# volatile disk - 3 - IO_volatile or bandwidth_volatile
def cluster_by_hand(data_cluster_first):
    labels = []
    count = [0, 0, 0, 0]
    for k in range(len(data_cluster_first)):
        if data_cluster_first[k][0] == 0:
            data_cluster_first[k][0] = 0.0001
        if data_cluster_first[k][1] == 0:
            data_cluster_first[k][1] = 0.0001
        IO_mul = data_cluster_first[k][2] / data_cluster_first[k][0]
        bandwidth_mul = data_cluster_first[k][3] / data_cluster_first[k][1]
        label = -1
        if IO_mul < IO_line and bandwidth_mul < bandwidth_line:
            label = 0
            count[0] += 1
        elif IO_mul >= IO_line and bandwidth_mul < bandwidth_line:
            label = 3
            count[3] += 1
        elif IO_mul < IO_line and bandwidth_mul >= bandwidth_line:
            label = 3
            count[3] += 1
        elif IO_mul >= IO_line and bandwidth_mul >= bandwidth_line:
            label = 3
            count[3] += 1
        labels.append(label)
    # print("count:", count[0], count[3])
    return labels

# accuracy of predicting stabledisk or volatile disk
def calculate_accuracy_first(items_predict, items_predict_labels_first):
    data = sum([[[row[6], row[7], row[8], row[9]]] for row in items_predict], [])
    labels_fact = cluster_by_hand(data)
    item_count = len(items_predict_labels_first)
    item_count_right = 0
    for i in range(item_count):
        if labels_fact[i] == items_predict_labels_first[i]:
            item_count_right += 1
    return (item_count_right / item_count)

# accuracy of predicting IOPS/bandwidth of stable/volatile disk
def calculate_accuracy_second(items_predict_second, items_predict_labels_second, centers_cluster_second, disk_label_first):
    count_total = len(items_predict_second)
    right_total = 0
    for i in range(count_total):
        if disk_label_first == 0:
            IO_fact = items_predict_second[i][0][6]
            bandwidth_fact = items_predict_second[i][0][7]
        elif disk_label_first == 3:
            IO_fact = items_predict_second[i][0][8]
            bandwidth_fact = items_predict_second[i][0][9]
        min_distance = 999999999
        fact_label = -1
        for j in range(cluster_K_second[disk_label_first]):
            now_distance = (IO_fact - centers_cluster_second[j][0])**2 + (bandwidth_fact - centers_cluster_second[j][1])**2
            if now_distance < min_distance:
                min_distance = now_distance
                fact_label = j
        if fact_label == items_predict_labels_second[i]:
            right_total += 1
    return float(right_total / count_total)

def plot_fenbu():
    data_temp = ''
    IO_multiple = []
    bandwidth_multiple = []
    with open(data_temp, "r") as f:
        for line in f:
            item = line.strip().split(',')
            item[1:6] = map(int, item[1:6])
            item[6:10] = map(float, item[6:10])
            item[10] = int(item[10])
            if item[6] == 0:
                item[6] = 0.0001
            if item[7] == 0:
                item[7] = 0.0001
            IO_mul = float(item[8] / item[6])
            bandwidth_mul = float(item[9] / item[7])
            if(IO_mul < 500 and bandwidth_mul < 75 and IO_mul != 0 and bandwidth_mul != 0):
                IO_multiple.append(IO_mul)
                bandwidth_multiple.append(bandwidth_mul)
    plt.figure()
    plt.scatter(IO_multiple, bandwidth_multiple, s=2)
    plt.xlabel("IO")
    plt.ylabel("bandwidth")
    plt.savefig(f"fenbu.png")
    plt.clf()

# items
#  (disk_ID, average_IOPS_predict, average_bandwidth_predict, disk_timestamp_num, disk_capacity, average_IOPS, average_bandwidth, item_source_file, peak_IOPS, peak_bandwidth, disk_label_first, disk_label_second)
def load_data_allocation(items_predict_second_result):
    items = []
    for i in range(len(items_predict_second_result)):
        disk_ID = items_predict_second_result[i][0][0]
        disk_label_first = int(items_predict_second_result[i][1])
        disk_label_second = int(items_predict_second_result[i][2])

        if disk_label_first == 3:
            IOPS_predict = float(IO_predict_center[1][disk_label_second][0])
            bandwidth_predict = float(IO_predict_center[1][disk_label_second][1])
        elif disk_label_first == 0:
            IOPS_predict = float(IO_predict_center[disk_label_first][disk_label_second][0])
            bandwidth_predict = float(IO_predict_center[disk_label_first][disk_label_second][1])
        
        disk_timestamp_num = int(items_predict_second_result[i][0][10])
        disk_capacity = int(items_predict_second_result[i][0][1])
        average_IOPS = float(items_predict_second_result[i][0][6])
        average_bandwidth = float(items_predict_second_result[i][0][7])
        peak_IOPS = float(items_predict_second_result[i][0][8])
        peak_bandwidth = float(items_predict_second_result[i][0][9])
        item_source_file = int(items_predict_second_result[i][0][11])
        items.append([disk_ID, IOPS_predict, bandwidth_predict, disk_timestamp_num, disk_capacity, average_IOPS, average_bandwidth, item_source_file, peak_IOPS, peak_bandwidth, disk_label_first, disk_label_second])
    return items

# MD policy with fitting
# item (disk_ID, IOPS_predict, bandwidth_predict, disk_timestamp_num, disk_capacity, average_IOPS, average_bandwidth, item_source_file, peak_IOPS, peak_bandwidth, disk_label_first)
def CDA_policy(warehouses, warehouses_allo, item, warehouses_burst_count, burst_map, popt_ave_IO, popt_ave_bandwidth, popt_peak_IO, popt_peak_bandwidth, warehouses_flag):
    action = -1
    min_manhatten = 4
    item_label_first = item[10]
    item_label_second = item[11]

    # stable disk
    if item_label_first == 0:
        min_manhatten = 4
        for i in range(warehouse_number):

            if warehouses_flag[i] != 1:
                continue

            ave_IO_fit = multivar_piecewise([warehouses_burst_count[i]], *popt_ave_IO)
            ave_bandwidth_fit = multivar_piecewise([warehouses_burst_count[i]], *popt_ave_bandwidth)
            peak_IO_fit = multivar_piecewise([warehouses_burst_count[i]], *popt_peak_IO)
            peak_bandwidth_fit = multivar_piecewise([warehouses_burst_count[i]], *popt_peak_bandwidth)
            
            if warehouses[i][0]+item[4]<=max_capacity[i]*reservation_rate and warehouses[i][1]+ave_IO_fit+item[1]<=max_IOPS[i]*reservation_rate and warehouses[i][2]+ave_bandwidth_fit+item[2]<=max_bandwidth[i]*reservation_rate and peak_IO_fit<=max_IOPS[i]*1 and peak_bandwidth_fit<=max_bandwidth[i]*1:
                w1 = (warehouses[i][0] + item[4]) / max_capacity[i]
                w2 = (warehouses[i][1] + ave_IO_fit + item[1]) / max_IOPS[i]
                w3 = (warehouses[i][2] + ave_bandwidth_fit + item[2]) / max_bandwidth[i]
                w4 = (w1 + w2 + w3) / 3
                now_manhatten = abs(w1 - w4) + abs(w2 - w4) + abs(w3 - w4)
                if now_manhatten < min_manhatten:
                    action = i
                    min_manhatten = now_manhatten
    # volatile disk
    elif item_label_first == 3:
        min_manhatten = 4
        for i in range(warehouse_number):

            if warehouses_flag[i] != 1:
                continue

            fit_X = warehouses_burst_count[i][:]
            fit_X[burst_map.index(item_label_second)] += 1
            
            ave_IO_fit = multivar_piecewise([fit_X], *popt_ave_IO)
            ave_bandwidth_fit = multivar_piecewise([fit_X], *popt_ave_bandwidth)
            peak_IO_fit = multivar_piecewise([fit_X], *popt_peak_IO)
            peak_bandwidth_fit = multivar_piecewise([fit_X], *popt_peak_bandwidth)

            if warehouses[i][0]+item[4]<=max_capacity[i]*reservation_rate and warehouses[i][1]+ave_IO_fit<=max_IOPS[i]*reservation_rate and warehouses[i][2]+ave_bandwidth_fit<=max_bandwidth[i]*reservation_rate and peak_IO_fit<=max_IOPS[i]*1 and peak_bandwidth_fit<=max_bandwidth[i]*1:
                w1 = (warehouses[i][0] + item[4]) / max_capacity[i]
                w2 = (warehouses[i][1] + ave_IO_fit) / max_IOPS[i]
                w3 = (warehouses[i][2] + ave_bandwidth_fit) / max_bandwidth[i]
                w4 = (w1 + w2 + w3) / 3
                now_manhatten = abs(w1 - w4) + abs(w2 - w4) + abs(w3 - w4)
                if now_manhatten < min_manhatten:
                    action = i
                    min_manhatten = now_manhatten
    return action

#  (disk_ID, average_IOPS_predict, average_bandwidth_predict, disk_timestamp_num, disk_capacity, average_IOPS, average_bandwidth, item_source_file, peak_IOPS, peak_bandwidth, disk_label_first, disk_label_second)
def calculate_items(item, action, warehouses_total_time, warehouses_flag):

    warehouse_violation_time = 0

    # Here are file paths of CD traces
    item_dir = ''
    item_dir_2 = ''

    disk_name = item[0]
    disk_capacity = item[4]
    item_source_file = item[7]
    disk_label_first = item[10]

    if item_source_file == 0:
        f = os.path.join(item_dir, str(disk_name))
    if item_source_file == 2:
        f = os.path.join(item_dir_2, str(disk_name))

    for i in range(evaluate_time_number):

        disk_IO_bandwidth = linecache.getline(f, i + 1)
        fields = disk_IO_bandwidth.strip().split(',')
        disk_IO = int(fields[1]) + int(fields[3])
        disk_bandwidth = int(fields[2]) + int(fields[4])

        warehouses_total_time[i][action][0] += disk_capacity
        if disk_label_first == 0:
            warehouses_total_time[i][action][1] += disk_IO
            warehouses_total_time[i][action][2] += disk_bandwidth
            if warehouses_total_time[i][action][1] > stable_disk_IO_evaluate[action][1]:
                stable_disk_IO_evaluate[action][0] = i + 1
                stable_disk_IO_evaluate[action][1] = warehouses_total_time[i][action][1]
            if warehouses_total_time[i][action][2] > stable_disk_bandwidth_evaluate[action][1]:
                stable_disk_bandwidth_evaluate[action][0] = i + 1
                stable_disk_bandwidth_evaluate[action][1] = warehouses_total_time[i][action][2]
        elif disk_label_first == 3:
            warehouses_total_time[i][action][3] += disk_IO
            warehouses_total_time[i][action][4] += disk_bandwidth
            if warehouses_total_time[i][action][3] > burst_disk_IO_evaluate[action][1]:
                burst_disk_IO_evaluate[action][0] = i + 1
                burst_disk_IO_evaluate[action][1] = warehouses_total_time[i][action][3]
            if warehouses_total_time[i][action][4] > burst_disk_bandwidth_evaluate[action][1]:
                burst_disk_bandwidth_evaluate[action][0] = i + 1
                burst_disk_bandwidth_evaluate[action][1] = warehouses_total_time[i][action][4]

        if warehouses_total_time[i][action][1] + warehouses_total_time[i][action][3] > max_IOPS[action] or warehouses_total_time[i][action][2] + warehouses_total_time[i][action][4]> max_bandwidth[action]:
            warehouse_violation_time += 1
    
    if warehouse_violation_time > max_violation_time_line:
        warehouses_flag[action] = -1
        print(f"warehouse {action} xxx")

    return 0

# (has been deprecated)
def calculate_warehouses(warehouses, items_chosen, time):
    # 0 capacity 1 2 stable_IO & bandwidth 3 4 burst IO & bandwidth
    warehouses_evaluate_stable_burst = [[0 for _ in range(5)] for _ in range(warehouse_number)]

    # Here are file paths of CD traces
    item_dir = ''
    item_dir_2 = ''

    burst_disk_IO_evaluate_now = [[0 for _ in range(2)] for _ in range(warehouse_number)]
    burst_disk_bandwidth_evaluate_now = [[0 for _ in range(2)] for _ in range(warehouse_number)]
    stable_disk_IO_evaluate_now = [[0 for _ in range(2)] for _ in range(warehouse_number)]
    stable_disk_bandwidth_evaluate_now = [[0 for _ in range(2)] for _ in range(warehouse_number)]

    # capacity
    for i in range(warehouse_number):
        warehouses_evaluate_stable_burst[i][0] = warehouses[i][0]

    # IOPS and bandwidth
    for i in range(len(items_chosen)):
        # items_chosen (disk_ID, action, disk_timestamp_num, item_source_file, disk_label_first, disk_label_second)
        disk_name = items_chosen[i][0]
        warehouse_index = items_chosen[i][1]
        disk_timestamp_num = items_chosen[i][2]
        item_source_file = items_chosen[i][3]
        disk_label_first = items_chosen[i][4]
        if item_source_file == 0:
            f = os.path.join(item_dir, str(disk_name))
        if item_source_file == 2:
            f = os.path.join(item_dir_2, str(disk_name))
        disk_index = (time * time_interval) % disk_timestamp_num
        if disk_index == 0:
            disk_index = disk_timestamp_num
        disk_IO_bandwidth = linecache.getline(f, disk_index)
        fields = disk_IO_bandwidth.strip().split(',')
        disk_IO = int(fields[1]) + int(fields[3])
        disk_bandwidth = int(fields[2]) + int(fields[4])

        if disk_label_first == 0:
            warehouses_evaluate_stable_burst[warehouse_index][1] += disk_IO
            warehouses_evaluate_stable_burst[warehouse_index][2] += disk_bandwidth
        if disk_label_first == 3:
            warehouses_evaluate_stable_burst[warehouse_index][3] += disk_IO
            warehouses_evaluate_stable_burst[warehouse_index][4] += disk_bandwidth

        if disk_label_first == 0:
            stable_disk_IO_evaluate_now[warehouse_index][1] += disk_IO
            stable_disk_bandwidth_evaluate_now[warehouse_index][1] += disk_bandwidth
        if disk_label_first == 3:
            burst_disk_IO_evaluate_now[warehouse_index][1] += disk_IO
            burst_disk_bandwidth_evaluate_now[warehouse_index][1] += disk_bandwidth

    # (time, predict_max)
    for i in range(warehouse_number):
        if stable_disk_IO_evaluate_now[i][1] > stable_disk_IO_evaluate[i][1]:
            stable_disk_IO_evaluate[i][0] = time
            stable_disk_IO_evaluate[i][1] = stable_disk_IO_evaluate_now[i][1]
        if stable_disk_bandwidth_evaluate_now[i][1] > stable_disk_bandwidth_evaluate[i][1]:
            stable_disk_bandwidth_evaluate[i][0] = time
            stable_disk_bandwidth_evaluate[i][1] = stable_disk_bandwidth_evaluate_now[i][1]
        if burst_disk_IO_evaluate_now[i][1] > burst_disk_IO_evaluate[i][1]:
            burst_disk_IO_evaluate[i][0] = time
            burst_disk_IO_evaluate[i][1] = burst_disk_IO_evaluate_now[i][1]
        if burst_disk_bandwidth_evaluate_now[i][1] > burst_disk_bandwidth_evaluate[i][1]:
            burst_disk_bandwidth_evaluate[i][0] = time
            burst_disk_bandwidth_evaluate[i][1] = burst_disk_bandwidth_evaluate_now[i][1]

    return warehouses_evaluate_stable_burst

# (has been deprecated)
def plot_utilize(utilize, utilize_sum, warehouses_allo, utilize_stable_burst):
    for n in range(warehouse_number):
        capacity_utilize = []
        IOPS_utilize = []
        IOPS_stable_utilize = []
        IOPS_burst_utilize = []
        bandwidth_utilize = []
        bandwidth_stable_utilize = []
        bandwidth_burst_utilize = []
        IOPS_utilize_average = []
        bandwidth_utilize_average =[]
        IOPS_utilize_average_allo = []
        bandwidth_utilize_average_allo = []
        time = []
        for i in range(len(utilize)):
            capacity_utilize.append(utilize[i][n][0])
            IOPS_utilize.append(utilize[i][n][1])
            bandwidth_utilize.append(utilize[i][n][2])
            IOPS_stable_utilize.append(utilize_stable_burst[i][n][0])
            bandwidth_stable_utilize.append(utilize_stable_burst[i][n][1])
            IOPS_burst_utilize.append(utilize_stable_burst[i][n][2])
            bandwidth_burst_utilize.append(utilize_stable_burst[i][n][3])
            IOPS_utilize_average.append(utilize_sum[n][1] / evaluate_time_number)
            bandwidth_utilize_average.append(utilize_sum[n][2] / evaluate_time_number)
            IOPS_utilize_average_allo.append(warehouses_allo[n][1] / max_IOPS[n])
            bandwidth_utilize_average_allo.append(warehouses_allo[n][2] / max_bandwidth[n])
            time.append(i)

        plt.subplot(2, 2, 2)
        plt.plot(time, IOPS_utilize, c='blue', label='total_IOPS')
        plt.plot(time, bandwidth_utilize, c='green', label='total_bandwidth')
        plt.legend(loc='best')
        plt.subplot(2, 2, 1)
        plt.plot(time, capacity_utilize, c='red', label='capacity')
        plt.plot(time, IOPS_utilize_average_allo, c='yellow', linestyle='-.', label='ave_IOPS')
        plt.plot(time, bandwidth_utilize_average_allo, c='black', linestyle='--', label='ave_bandwidth')
        plt.legend(loc='best')
        plt.ylabel("utilize")
        plt.subplot(2, 2, 3)
        plt.plot(time, IOPS_stable_utilize, c='blue', label='stable_IOPS')
        plt.plot(time, bandwidth_stable_utilize, c='green', label='stable_bandwidth')
        plt.xlabel("time")
        plt.ylabel("utilize")
        plt.legend(loc='best')
        plt.subplot(2, 2, 4)
        plt.plot(time, IOPS_burst_utilize, c='blue', label='burst_IOPS')
        plt.plot(time, bandwidth_burst_utilize, c='green', label='burst_bandwidth')
        plt.xlabel("time")
        plt.legend(loc='best')
        # plt.plot(time, IOPS_utilize_average, c='yellow', label='ave_IOPS')
        # plt.plot(time, bandwidth_utilize_average, c='black', label='ave_bandwidth')
        plt.suptitle(f"stbu-warehouse{n+1}")
        plt.savefig(f"stbu_warehouse{n+1}.png")
        plt.clf()

    return 0

# main
if __name__ == "__main__":

    # plot_fenbu()
    
    for e in range(episodes):

        burst_disk_IO_evaluate = [[0 for _ in range(2)] for _ in range(warehouse_number)]
        burst_disk_bandwidth_evaluate = [[0 for _ in range(2)] for _ in range(warehouse_number)]
        stable_disk_IO_evaluate = [[0 for _ in range(2)] for _ in range(warehouse_number)]
        stable_disk_bandwidth_evaluate = [[0 for _ in range(2)] for _ in range(warehouse_number)]
        burst_disk_IO_predict_peak = [0 for _ in range(warehouse_number)]
        burst_disk_bandwidth_predict_peak = [0 for _ in range(warehouse_number)]
        burst_disk_IO_predict_ave = [0 for _ in range(warehouse_number)]
        burst_disk_bandwidth_predict_ave = [0 for _ in range(warehouse_number)]
        stable_disk_IO_predict = [0 for _ in range(warehouse_number)]
        stable_disk_bandwidth_predict = [0 for _ in range(warehouse_number)]

        # load data
        # items_cluster
        #  0 (disk_ID),
        #  1:6 (disk_capacity, disk_if_local, disk_type, vm_cpu, vm_mem),
        #  6:10 (average IOPS, average bandwidth, peak IOPS, peak bandwidth),
        #  10 (disk_timestamp_num),
        #  11 (item_source_file),
        #  12: (IOPS_bandwidth_with_time[])
        items_train = load_data_train()
        print(len(items_train))
        items_predict = load_data_predict()
        print(len(items_predict))

        # classifier(average IOPS, average bandwidth, peak_IO, peak_bandwidth)
        # classify
        data_cluster_first = sum([[[row[6], row[7], row[8], row[9]]] for row in items_train], [])
        labels_cluster_first = cluster_by_hand(data_cluster_first)
        items_cluster_first_result = [[x, y] for x, y in zip(items_train, labels_cluster_first)]
        # train
        data_train_first_X = sum([[[row[1], row[2], row[3], row[4], row[5]]] for row in items_train], [])
        data_train_first_Y = labels_cluster_first
        model_classify_first = DecisionTreeClassifier(criterion="gini")
        model_classify_first.fit(data_train_first_X, data_train_first_Y)
        # predict
        data_predict_first = sum([[[row[1], row[2], row[3], row[4], row[5]]] for row in items_predict], [])
        items_predict_labels_first = model_classify_first.predict(data_predict_first)
        items_predict_first_result = [[x, y] for x, y in zip(items_predict, items_predict_labels_first)]
        # calculate accuracy
        ten_fold_scores_first = cross_val_score(model_classify_first, data_train_first_X, data_train_first_Y, cv=10, scoring='accuracy')
        print("first - decision tree - train set - accuracy:",ten_fold_scores_first)
        print("first - decision tree - validate set - accuracy:", calculate_accuracy_first(items_predict, items_predict_labels_first))

        # predictor
        # 0 IO_stable - bandwidth_stable (ave_IO,  ave_bandwidth)
        # 3 IO_burst  - bandwidth_stable (peak_IO, ave_bandwidth)
        # 3 IO_stable - bandwidth_burst  (ave_IO,  peak_bandwidth)
        # 3 IO_burst  - bandwidth_burst  (peak_IO, peak_bandwidth)
        items_cluster_second_result = []
        items_predict_second_result = []

        burst_map = [0 for _ in range(cluster_K_second[3])]

        for i in range(cluster_first_dimension):

            if i == 1:
                continue
            if i == 2:
                continue
                
            # cluster
            items_cluster_second = sum([[row] for row in items_cluster_first_result if row[1] == i], [])
            if i == 0:
                data_cluster_second = sum([[[x[0][6], x[0][7]]] for x in items_cluster_second], [])
            elif i == 1:
                data_cluster_second = sum([[[x[0][8], x[0][7]]] for x in items_cluster_second], [])
            elif i == 2:
                data_cluster_second = sum([[[x[0][6], x[0][9]]] for x in items_cluster_second], [])
            elif i == 3:
                data_cluster_second = sum([[[x[0][8], x[0][9]]] for x in items_cluster_second], [])
            model_cluster_second = KMeans(n_clusters = cluster_K_second[i])
            model_cluster_second.fit(data_cluster_second)
            centers_cluster_second = model_cluster_second.cluster_centers_
            IO_predict_center.append(centers_cluster_second)
            labels_cluster_second = model_cluster_second.labels_
            items_cluster_second_result_mid = [x + [y] for x, y in zip(items_cluster_second, labels_cluster_second)]
            items_cluster_second_result = items_cluster_second_result + items_cluster_second_result_mid

            if i == 3:
                sorted_centers = sorted(enumerate(centers_cluster_second), key=lambda x: x[1][0])
                burst_map = [x[0] for x in sorted_centers][::-1]

            # calculate silhouette_score
            score = silhouette_score(data_cluster_second, labels_cluster_second)
            print(f"second - {i} - the average silhouette score is:", score)
            # 
            unique_labels_second, counts_second = np.unique(labels_cluster_second, return_counts=True)
            print(f"second - {i} - ", unique_labels_second, counts_second)
            print(centers_cluster_second)

            # train
            data_train_second_X = sum([[[x[0][1], x[0][2], x[0][3], x[0][4], x[0][5]]] for x in items_cluster_second], [])
            data_train_second_Y = labels_cluster_second
            model_classify_second = DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_split=10)
            model_classify_second.fit(data_train_second_X, data_train_second_Y)
            # predict
            items_predict_second = sum([[row] for row in items_predict_first_result if row[1] == i], [])
            data_predict_second = sum([[[x[0][1], x[0][2], x[0][3], x[0][4], x[0][5]]] for x in items_predict_second], [])
            items_predict_labels_second = model_classify_second.predict(data_predict_second)
            items_predict_second_result_mid = [x + [y] for x, y in zip(items_predict_second, items_predict_labels_second)]
            items_predict_second_result = items_predict_second_result + items_predict_second_result_mid
            # calculate accuracy
            ten_fold_scores_second = cross_val_score(model_classify_second, data_train_second_X, data_train_second_Y, cv=10, scoring='accuracy')
            print(f"second - {i} - decision tree - train set - accuracy:",ten_fold_scores_second)
            # 
            print(f"second - {i} - decision tree - validate set - accuracy:", calculate_accuracy_second(items_predict_second, items_predict_labels_second, centers_cluster_second, i))
        
        # fitting
        fit_data_X = load_data_fit(0)
        fit_data_ave_IO = load_data_fit(1)
        fit_data_ave_bandwidth = load_data_fit(2)
        fit_data_peak_IO = load_data_fit(3)
        fit_data_peak_bandwidth = load_data_fit(4)

        popt_ave_IO, pcov_ave_IO = curve_fit(multivar_piecewise, fit_data_X, fit_data_ave_IO)
        print("popt_ave_IO:",popt_ave_IO)
        popt_ave_bandwidth, pcov_ave_bandwidth = curve_fit(multivar_piecewise, fit_data_X, fit_data_ave_bandwidth)
        print("popt_ave_bandwidth:",popt_ave_bandwidth)
        popt_peak_IO, pcov_peak_IO = curve_fit(multivar_piecewise, fit_data_X, fit_data_peak_IO)
        print("popt_peak_IO:",popt_peak_IO)
        popt_peak_bandwidth, pcov_peak_bandwidth = curve_fit(multivar_piecewise, fit_data_X, fit_data_peak_bandwidth)
        print("popt_peak_bandwidth:",popt_peak_bandwidth)

        r_ave_IO = np.corrcoef(fit_data_ave_IO, multivar_piecewise(fit_data_X, *popt_ave_IO))
        print("r_ave_IO:", r_ave_IO)
        r_ave_bandwidth = np.corrcoef(fit_data_ave_bandwidth, multivar_piecewise(fit_data_X, *popt_ave_bandwidth))
        print("r_ave_bandwidth:", r_ave_bandwidth)
        r_peak_IO = np.corrcoef(fit_data_peak_IO, multivar_piecewise(fit_data_X, *popt_peak_IO))
        print("r_peak_IO:", r_peak_IO)
        r_peak_bandwidth = np.corrcoef(fit_data_peak_bandwidth, multivar_piecewise(fit_data_X, *popt_peak_bandwidth))
        print("r_peak_bandwidth:", r_peak_bandwidth)

        # CDP policy
        # (capacity, stable_IO, stable_bandwidth, burst_IO, burst_bandwidth)
        warehouses = [[0 for _ in range(5)] for _ in range(warehouse_number)]
        warehouses_allo = [[0 for _ in range(3)] for _ in range(warehouse_number)]
        warehouses_evaluate = [[0 for _ in range(3)] for _ in range(warehouse_number)]
        warehouses_evaluate_stable_burst = [[0 for _ in range(5)] for _ in range(warehouse_number)]

        warehouses_total_time = [[[0 for _ in range(5)] for _ in range(warehouse_number)] for _ in range(evaluate_time_number)]
        warehouses_flag = [1, 1, 1, 1, 1, 1, 1, 1, 1]

        # various_kinds_volatile_disk_num
        burst_disk_IO_fit = [0 for _ in range(warehouse_number)]
        burst_disk_bandwidth_fit = [0 for _ in range(warehouse_number)]
        warehouses_burst_count = [[0 for _ in range(cluster_K_second[3])] for _ in range(warehouse_number)]
        warehouses_stable_count = [[0 for _ in range(cluster_K_second[0])] for _ in range(warehouse_number)]
        # load_imb
        average_dimension_load_imb = 0
        average_warehouse_load_imb = 0
        # violation_time and max_violation_duration
        violation_times = 0
        max_violation_time = 0
        violation_time = 0
        violation_times_warehouses = [0 for _ in range(warehouse_number)]
        max_violation_time_warehouses = [0 for _ in range(warehouse_number)]
        violation_time_warehouses = [0 for _ in range(warehouse_number)]
        # item (disk_ID, average_IOPS_predict, average_bandwidth_predict, disk_timestamp_num, disk_capacity, average_IOPS, average_bandwidth, item_source_file, peak_IOPS, peak_bandwidth, disk_label_first, disk_label_second)
        items = load_data_allocation(items_predict_second_result)
        # items_chosen (disk_ID, action, disk_timestamp_num, item_source_file, disk_label_first, disk_label_second)
        items_chosen = []
        # 
        step = 0
        # 
        utilize = []
        utilize_stable_burst = []
        # 
        stop_time_now = 0

        while len(items) > episode_remain:
            # item (disk_ID, IOPS_predict, bandwidth_predict, disk_timestamp_num, disk_capacity, average_IOPS, average_bandwidth, item_source_file, peak_IOPS, peak_bandwidth, disk_label_first)
            item = random.choice(items)
            # choose action
            action = CDA_policy(warehouses, warehouses_allo, item, warehouses_burst_count, burst_map, popt_ave_IO, popt_ave_bandwidth, popt_peak_IO, popt_peak_bandwidth, warehouses_flag)
            # check
            if action == -1:
                stop_time_now += 1
                if stop_time_now == stop_times:
                    print("step:", step)
                    print("item: ", item[10], item[11], item[4], item[1], item[2], item[5], item[6], item[8], item[9])
                    break
                continue
            # update
            if item[10] == 0:
                warehouses[action][0] += item[4]
                warehouses[action][1] += item[1]
                warehouses[action][2] += item[2]
            elif item[10] == 3:
                warehouses[action][0] += item[4]
                if item[1] > warehouses[action][3]:
                    warehouses[action][3] = item[1]
                if item[2] > warehouses[action][4]:
                    warehouses[action][4] = item[2]
                
                burst_disk_IO_fit[action] += item[5]
                burst_disk_bandwidth_fit[action] += item[6]
            
            warehouses_allo[action][0] += item[4]
            warehouses_allo[action][1] += item[5]
            warehouses_allo[action][2] += item[6]

            calculate_items(item, action, warehouses_total_time, warehouses_flag)

            if item[10] == 3:
                warehouses_burst_count[action][burst_map.index(item[11])] += 1
            elif item[10] == 0:
                warehouses_stable_count[action][item[11]] += 1
            # update
            step += 1
            if step % 500 == 0:
                print("items_num:",step)
            items_chosen.append([item[0], action, item[3], item[7], item[10], item[11]])
            items.remove(item)

        print("warehouses_burst_count:", warehouses_burst_count)
        print("warehouses_stable_count", warehouses_stable_count)
        
        # calculate resource_utilization
        total_utilize = [0, 0, 0]
        for i in range(warehouse_number):
            total_utilize[0] += warehouses_allo[i][0] / (max_capacity[i] * warehouse_number)
            total_utilize[1] += warehouses_allo[i][1] / (max_IOPS[i] * warehouse_number)
            total_utilize[2] += warehouses_allo[i][2] / (max_bandwidth[i] * warehouse_number)
        print("total_utilize:", total_utilize)

        # 
        for i in range(warehouse_number):
            stable_disk_IO_predict[i] = warehouses[i][1]
            stable_disk_bandwidth_predict[i] =warehouses[i][2]
            burst_disk_IO_predict_peak[i] = multivar_piecewise([warehouses_burst_count[i]], *popt_peak_IO)
            burst_disk_bandwidth_predict_peak[i] = multivar_piecewise([warehouses_burst_count[i]], *popt_peak_bandwidth)
            burst_disk_IO_predict_ave[i] = multivar_piecewise([warehouses_burst_count[i]], *popt_ave_IO)
            burst_disk_bandwidth_predict_ave[i] = multivar_piecewise([warehouses_burst_count[i]], *popt_ave_bandwidth)
        print("stable_IO_predict:",stable_disk_IO_predict)
        print("stable_bandwidth_predict:",stable_disk_bandwidth_predict)
        print("burst_IO_predict_peak:",burst_disk_IO_predict_peak)
        print("burst_bandwidth_predict_peak:",burst_disk_bandwidth_predict_peak)
        print("burst_IO_predict_ave:",burst_disk_IO_predict_ave)
        print("burst_bandwidth_predict_ave:",burst_disk_bandwidth_predict_ave)

        # evaluate     
        warehouse_dimension_utilize_sum = [[0 for _ in range(3)] for _ in range(warehouse_number)]
        for i in range(evaluate_time_number):
            # evaluate
            # warehouses_evaluate_stable_burst = calculate_warehouses(warehouses, items_chosen, i + 1)
            warehouses_evaluate_stable_burst = warehouses_total_time[i]
            for k in range(warehouse_number):
                    warehouses_evaluate[k][0] = warehouses_evaluate_stable_burst[k][0]
                    warehouses_evaluate[k][1] = warehouses_evaluate_stable_burst[k][1] + warehouses_evaluate_stable_burst[k][3]
                    warehouses_evaluate[k][2] = warehouses_evaluate_stable_burst[k][2] + warehouses_evaluate_stable_burst[k][4]
            # 
            warehouse_array = np.array(warehouses_evaluate)
            warehouse_array_stable_burst = np.array(warehouses_evaluate_stable_burst)
            warehouse_dimension_utilize = [[0 for _ in range(3)] for _ in range(warehouse_number)]
            warehouse_dimension_utilize_stable_burst = [[0 for _ in range(4)] for _ in range(warehouse_number)]

            for k in range(warehouse_number):
                for j in range(3):
                    warehouse_dimension_utilize[k][j] = warehouse_array[k][j] / warehouse_max[j][k]

            for k in range(warehouse_number):
                # stable IO stable bandwidth burst IO burst bandwidth
                warehouse_dimension_utilize_stable_burst[k][0] = warehouse_array_stable_burst[k][1] / warehouse_max[1][k]
                warehouse_dimension_utilize_stable_burst[k][1] = warehouse_array_stable_burst[k][2] / warehouse_max[2][k]
                warehouse_dimension_utilize_stable_burst[k][2] = warehouse_array_stable_burst[k][3] / warehouse_max[1][k]
                warehouse_dimension_utilize_stable_burst[k][3] = warehouse_array_stable_burst[k][4] / warehouse_max[2][k]

            warehouse_dimension_utilize_sum += warehouse_dimension_utilize
            utilize.append([warehouse_dimension_utilize[i] for i in range(warehouse_number)])
            utilize_stable_burst.append([warehouse_dimension_utilize_stable_burst[i] for i in range(warehouse_number)])
            if i % 2000 == 0:
                print("i:",i)

            # violation
            violation_flag = 0
            violation_flag_warehouses = [0 for _ in range(warehouse_number)]
            # 
            for w in range(warehouse_number):
                for i in range(3):
                    if warehouse_dimension_utilize[w][i] > 1:
                        violation_flag = 1
                        violation_flag_warehouses[w] = 1

            for m in range(warehouse_number):
                if violation_flag_warehouses[m] == 1:
                    violation_times_warehouses[m] += 1
                    violation_time_warehouses[m] += 1
                    if violation_time_warehouses[m] > max_violation_time_warehouses[m]:
                        max_violation_time_warehouses[m] = violation_time_warehouses[m]
                else:
                    if violation_time_warehouses[m] != 0:
                        if violation_time_warehouses[m] > max_violation_time_warehouses[m]:
                            max_violation_time_warehouses[m] = violation_time_warehouses[m]
                        violation_time_warehouses[m] = 0

            # load_imb
            # dimensions(capacity,IOPS,bandwidth)(has been deprecated)
            dimension_utilize_var = np.var(warehouse_dimension_utilize, 1)
            dimension_utilize_std = np.std(warehouse_dimension_utilize, 1)
            dimension_utilize_mean = np.mean(warehouse_dimension_utilize, 1)
            if 0 in dimension_utilize_mean:
                dimension_utilize_mean[np.where(dimension_utilize_mean == 0)] = 0.00001
            # warehouses
            warehouse_utilize_var = np.var(warehouse_dimension_utilize, 0)
            warehouse_utilize_std = np.std(warehouse_dimension_utilize, 0)
            warehouse_utilize_mean = np.mean(warehouse_dimension_utilize, 0)
            if 0 in warehouse_utilize_mean:
                warehouse_utilize_mean[np.where(warehouse_utilize_mean == 0)] = 0.00001
            dimension_load_imb = dimension_utilize_std / dimension_utilize_mean
            warehouse_load_imb = warehouse_utilize_std / warehouse_utilize_mean

            # load_imb
            average_dimension_load_imb += dimension_load_imb
            average_warehouse_load_imb += warehouse_load_imb

        print("stable_IO_evaluate:", stable_disk_IO_evaluate)
        print("stable_bandwidth_evaluate:", stable_disk_bandwidth_evaluate)
        print("burst_IO_evaluate_peak:", burst_disk_IO_evaluate)
        print("burst_bandwidth_evaluate_peak:", burst_disk_bandwidth_evaluate)
        print("burst_IO_evaluate_ave:", burst_disk_IO_fit)
        print("burst_bandwidth_evaluate_ave:", burst_disk_bandwidth_fit)

        # train episode
        print("episode:", e)    

        # ave_load_imb
        average_dimension_load_imb = average_dimension_load_imb / evaluate_time_number
        average_warehouse_load_imb = average_warehouse_load_imb / evaluate_time_number

        max_violation_time = max(max_violation_time_warehouses)
        violation_times = sum(violation_times_warehouses)
        ave_total_utilize = np.mean(total_utilize)
        ave_average_dimension_load_imb = np.mean(average_dimension_load_imb)
        ave_average_warehouse_load_imb = np.mean(average_warehouse_load_imb)
        
        # Here is the file path to record the experiment result
        result_file = ''
        with open(result_file, 'a') as f:
            f.write("\n")
            f.write("item_num:" + str(len(items_chosen)) + "\n")
            f.write("dimension_load_imb:" + str(ave_average_dimension_load_imb) + "\n")
            f.write("warehouses_load_imb:" + str(ave_average_warehouse_load_imb) + "\n")
            f.write("violation_times:" + str(violation_times) + "\n")
            f.write("max_violation_time:" + str(max_violation_time) + "\n")
            f.write("ave_utilize:" + str(ave_total_utilize) + "\n")
            
            f.write("min_timestamp_num:" + str(min_timestamp_num) + "  time_interval:" + str(time_interval) + "  evaluate_time:" + str(evaluate_time_number) + "\n")
            f.write("stop_time_numbers:" + str(stop_times) + "  IO_line:" + str(IO_line) + "  bandwidth_line:" + str(bandwidth_line) + "\n")
            f.write("average_dimension_load_imb:" + str(average_dimension_load_imb) + "\n")
            f.write("average_warehouse_load_imb:" + str(average_warehouse_load_imb) + "\n")
            f.write("warehouses_violation_times:" + str(violation_times_warehouses) + "\n")
            f.write("warehouses_max_violation_time:" + str(max_violation_time_warehouses) + "\n")
            f.write("total_utilize:"+str(total_utilize)+"\n")
            f.write("stable_IO_predict:"+str(stable_disk_IO_predict)+"\n")
            f.write("stable_bandwidth_predict:"+str(stable_disk_bandwidth_predict)+"\n")
            f.write("burst_IO_predict_peak:"+str(burst_disk_IO_predict_peak)+"\n")
            f.write("burst_bandwidth_predict_peak:"+str(burst_disk_bandwidth_predict_peak)+"\n")
            f.write("stable_IO_evaluate:"+str(stable_disk_IO_evaluate)+"\n")
            f.write("stable_bandwidth_evaluate:"+str(stable_disk_bandwidth_evaluate)+"\n")
            f.write("burst_IO_evaluate:"+str(burst_disk_IO_evaluate)+"\n")
            f.write("burst_bandwidth_evaluate:"+str(burst_disk_bandwidth_evaluate)+"\n\n")
        

        # 
        plot_utilize(utilize, warehouse_dimension_utilize_sum, warehouses_allo, utilize_stable_burst)
        