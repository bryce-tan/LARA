import numpy as np

warehouse_number = 9

max_capacity = [170000, 170000, 170000, 170000, 170000, 170000, 50000, 50000, 50000]
max_IOPS = [400000, 400000, 400000, 250000, 250000, 250000, 500000, 500000, 500000] 
max_bandwidth = [200000, 200000, 200000, 600000, 600000, 600000, 600000, 600000, 600000]

warehouse_max = np.array([max_capacity, max_IOPS, max_bandwidth])

evaluate_time_number = 10000