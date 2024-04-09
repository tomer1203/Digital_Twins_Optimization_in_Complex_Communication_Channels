import numpy as np
from matplotlib import pyplot as plt
import datetime
import matplotlib
time_lst = np.load(".\\outputs\\time_lst.npy",allow_pickle=True)
physfad_time_lst = np.load(".\\outputs\\physfad_time_lst.npy",allow_pickle=True)
zogd_time_lst = np.load(".\\outputs\\zogd_time_lst.npy",allow_pickle=True)
rand_search_time_lst = np.load(".\\outputs\\rand_search_time_lst.npy",allow_pickle=True)
dnn_physfad_capacity_lst = np.load(".\\outputs\\dnn_physfad_capacity_lst.npy",allow_pickle=True)
physfad_capacity = np.load(".\\outputs\\physfad_capacity.npy",allow_pickle=True)
zogd_capacity = np.load(".\\outputs\\zogd_capacity.npy",allow_pickle=True)
random_search_capacity = np.load(".\\outputs\\random_search_capacity.npy",allow_pickle=True)
plots_lst = []
legend_lst = []
font = {'size'   : 14}

matplotlib.rc('font', **font)
if time_lst is not None:
    delta_time_list = [(t-time_lst[0])/ datetime.timedelta(seconds=1) for t in time_lst]
    delta_time_list[0] = delta_time_list[0] + 0.01
    best_capacity = 0
    number_of_consecutive_non_increases = 0
    best_iteration = 0
    for i,capacity in enumerate(dnn_physfad_capacity_lst):
        number_of_consecutive_non_increases = number_of_consecutive_non_increases + 1
        if capacity>best_capacity:
            best_capacity = capacity
            number_of_consecutive_non_increases = 0
        best_iteration = i
        if number_of_consecutive_non_increases > 40:
            best_iteration = i-40
            break
    p1, = plt.plot(delta_time_list,dnn_physfad_capacity_lst,'-',color=plt.rcParams['axes.prop_cycle'].by_key()['color'][2])
    p1.set_label("DNN model")
    plots_lst.append(p1)
    legend_lst.append("DNN model")
if physfad_time_lst is not None:
    physfad_delta_time_list = (physfad_time_lst-physfad_time_lst[0])/ datetime.timedelta(seconds=1)

    physfad_delta_time_list[0] = physfad_delta_time_list[0]+0.01
    max_index = len(physfad_delta_time_list[physfad_delta_time_list<25])-1
    p3, = plt.plot(physfad_delta_time_list[:max_index], physfad_capacity[:max_index],'--',color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
    p3.set_label("Physfad optimization")
    plots_lst.append(p3)
    legend_lst.append("Physfad optimization")
if zogd_time_lst is not None:
    # zogd_delta_time_list = [(t - zogd_time_lst[0]) / datetime.timedelta(seconds=1) for t in zogd_time_lst]
    zogd_delta_time_list = (zogd_time_lst-zogd_time_lst[0])/ datetime.timedelta(seconds=1)
    zogd_delta_time_list[0] = zogd_delta_time_list[0] + 0.01
    max_index = len(zogd_delta_time_list[zogd_delta_time_list<25])-1
    p4, = plt.plot(zogd_delta_time_list[:max_index], zogd_capacity[:max_index],'-.s',color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
    p4.set_label("Zero Order Gradient Descent")
    plots_lst.append(p4)
    legend_lst.append("Zero Order Gradient Descent")
if rand_search_time_lst is not None:
    random_delta_time_list = (rand_search_time_lst-rand_search_time_lst[0]) / datetime.timedelta(seconds=1)
    random_delta_time_list[0] = random_delta_time_list[0] + 0.01
    max_index = len(random_delta_time_list[random_delta_time_list<25])-1

    random_capacity = [max(random_search_capacity[:i+1]) for i,x in enumerate(random_search_capacity)]
    p5, = plt.plot(random_delta_time_list[:max_index], random_capacity[:max_index],':',color=plt.rcParams['axes.prop_cycle'].by_key()['color'][3],linewidth=2)
    p5.set_label("Random Search")
    plots_lst.append(p5)
    legend_lst.append("Random Search")
plt.legend(plots_lst,legend_lst)
plt.xlabel("seconds[s]")
plt.ylabel("Rate[Bits/Channel used]")
plt.xscale('log')

plt.show()