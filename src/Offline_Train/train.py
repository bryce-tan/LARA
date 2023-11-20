import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from scipy.optimize import curve_fit

import sys
sys.path.append('.')

from load_data.load import load_data_update_model
from train_conf import multivar_piecewise
from train_conf import model_classify_first
from train_conf import model_classify_second_stable
from train_conf import model_classify_second_volatile
from train_conf import IO_line
from train_conf import bandwidth_line
from train_conf import cluster_K_stable
from train_conf import cluster_K_volatile

# stable disk or wolatile disk
# stable disk - 0 - IO_stable and bandwidth_stable
# volatile disk - 3 - IO_volatile or bandwidth_volatile
# [0:4] (average_IOPS, average_bandwidth, peak_IO, peak_bandwidth)
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
    return labels

def train_predictor_stable(items_cluster_second):

    # [0][1:6] (disk_capacity, disk_if_local, disk_type, vm_cpu, vm_mem)
    # [0][6:10] (average_IOPS, average_bandwidth, peak_IO, peak_bandwidth)

    data_cluster_second = sum([[[x[0][6], x[0][7]]] for x in items_cluster_second], [])
    model_cluster_second = KMeans(n_clusters = cluster_K_stable)
    model_cluster_second.fit(data_cluster_second)
    labels_cluster_second = model_cluster_second.labels_

    global centers_cluster_second_stable
    centers_cluster_second_stable = model_cluster_second.cluster_centers_

    data_train_second_X = sum([[[x[0][1], x[0][2], x[0][3], x[0][4], x[0][5]]] for x in items_cluster_second], [])
    data_train_second_Y = labels_cluster_second
    model_classify_second_stable.fit(data_train_second_X, data_train_second_Y)

    return

def train_predictor_volatile(items_cluster_second):

    # [0][1:6] (disk_capacity, disk_if_local, disk_type, vm_cpu, vm_mem)
    # [0][6:10] (average_IOPS, average_bandwidth, peak_IO, peak_bandwidth)

    data_cluster_second = sum([[[x[0][8], x[0][9]]] for x in items_cluster_second], [])
    model_cluster_second = KMeans(n_clusters = cluster_K_volatile)
    model_cluster_second.fit(data_cluster_second)
    labels_cluster_second = model_cluster_second.labels_

    centers_cluster_second = model_cluster_second.cluster_centers_
    sorted_centers = sorted(enumerate(centers_cluster_second), key=lambda x: x[1][0])

    global burst_map
    burst_map = [x[0] for x in sorted_centers][::-1]

    data_train_second_X = sum([[[x[0][1], x[0][2], x[0][3], x[0][4], x[0][5]]] for x in items_cluster_second], [])
    data_train_second_Y = labels_cluster_second
    model_classify_second_volatile.fit(data_train_second_X, data_train_second_Y)

    return

def train_estimator(fit_data):

    fit_data_X = fit_data[0]
    fit_data_ave_IO = fit_data[1]
    fit_data_ave_bandwidth = fit_data[2]
    fit_data_peak_IO = fit_data[3]
    fit_data_peak_bandwidth = fit_data[4]

    global popt_ave_IO, popt_ave_bandwidth, popt_peak_IO, popt_peak_bandwidth

    popt_ave_IO, pcov_ave_IO = curve_fit(multivar_piecewise, fit_data_X, fit_data_ave_IO)
    popt_ave_bandwidth, pcov_ave_bandwidth = curve_fit(multivar_piecewise, fit_data_X, fit_data_ave_bandwidth)
    popt_peak_IO, pcov_peak_IO = curve_fit(multivar_piecewise, fit_data_X, fit_data_peak_IO)
    popt_peak_bandwidth, pcov_peak_bandwidth = curve_fit(multivar_piecewise, fit_data_X, fit_data_peak_bandwidth)

    return

# train classifier & train predictor
def train_classifier(items_train):
    # [1:6] (disk_capacity, disk_if_local, disk_type, vm_cpu, vm_mem)
    # [6:10] (average_IOPS, average_bandwidth, peak_IO, peak_bandwidth)
    # classify
    data_cluster_first = sum([[[row[6], row[7], row[8], row[9]]] for row in items_train], [])
    labels_cluster_first = cluster_by_hand(data_cluster_first)
    items_cluster_first_result = [[x, y] for x, y in zip(items_train, labels_cluster_first)]
    # train classifier
    data_train_first_X = sum([[[row[1], row[2], row[3], row[4], row[5]]] for row in items_train], [])
    data_train_first_Y = labels_cluster_first
    model_classify_first.fit(data_train_first_X, data_train_first_Y)

    # train predictor
    items_cluster_second_stable = sum([[row] for row in items_cluster_first_result if row[1] == 0], [])
    train_predictor_stable(items_cluster_second_stable)
    items_cluster_second_volatile = sum([[row] for row in items_cluster_first_result if row[1] == 3], [])
    train_predictor_volatile(items_cluster_second_volatile)

    return

def update_model():

    items = load_data_update_model()
    train_classifier(items)

    return