import datetime
import cProfile,pstats,io
from matplotlib import pyplot as plt
import torch
import numpy as np
from ChannelMatrixEvaluation import test_configurations_capacity,fill_ris_config,get_configuration_parameters
from ChannelMatrixEvaluation import capacity_loss
from PhysFadPy import get_bessel_w
def print_train_iteration(start_time,batch_idx,epoch,max_epochs,train_ldr,pr,NMSE_lst):
    print("progress: "+str(100*(batch_idx/len(train_ldr))).split(".")[0]+"%"+" Train NMSE: {0:.4f}".format(np.mean(NMSE_lst)))
    time_elapsed = datetime.datetime.now() - start_time
    time_per_iteration = time_elapsed / (batch_idx + 1)
    completed = epoch/max_epochs +(batch_idx/len(train_ldr))/max_epochs
    time_left = time_per_iteration*len(train_ldr)*(max_epochs-epoch-1)+time_per_iteration * (len(train_ldr) - batch_idx - 1)
    print("completed {0:.2%} time left: {1}".format(completed, str(time_left).split(".")[0]))
    if pr is not None and batch_idx == 40:
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        pr.dump_stats("train_time_profile.prof")
        exit()
def plot_train_epoch(Y,oupt,model_output_capacity):
    plt.plot(torch.std(Y, 0).t().cpu().detach().numpy(), 'b--')
    plt.plot(torch.std(oupt, 0).t().cpu().detach().numpy(), "r")
    plt.legend(["ground truth variation", "output variation"])
    plt.show()
    plt.plot((torch.std(oupt, 0) / (torch.std(Y, 0))).t().cpu().detach().numpy(), "g--")
    plt.title("output.std/GT.std()")
    plt.show()
    if not model_output_capacity:
        plt.plot(abs(Y[0, :]).t().cpu().detach().numpy(), 'b--')
        plt.plot(abs(oupt[0, :]).t().cpu().detach().numpy(), 'r')
        plt.legend(["ground truth", "output"])
    plt.show()
def test_dnn_optimization(opt_inp_lst,device,noise):
    parameters = get_configuration_parameters(device)
    # while (loss.item() > -10000 and iters < num_of_iterations):
    (freq, x_tx, y_tx, _, _, _, x_rx, y_rx, _, _, _, x_env, y_env, _, _, _, x_ris_c, y_ris_c) = parameters
    W = get_bessel_w(freq, x_tx, y_tx, x_rx, y_rx, x_env, y_env, x_ris_c, y_ris_c, device)
    physfad_model_capacity_lst = []
    for i, opt_inp in enumerate(opt_inp_lst):
        physfad_capacity, physfad_H = test_configurations_capacity(parameters,
                                                                   fill_ris_config(opt_inp ** 2, device), W,
                                                                   device,noise=noise)
        # physfad_capacity = torch.sum(torch.abs(physfad_H))
        print("iter {0} physfad capacity {1}".format(20 * i, physfad_capacity))
        physfad_model_capacity_lst.append(physfad_capacity)
    np_dnn_physfad_capacity = np.array([x.detach().numpy() for x in physfad_model_capacity_lst])
    return np_dnn_physfad_capacity
def plot_model_optimization(time_lst,physfad_model_capacity_lst,model_capacity_lst,
                            physfad_time_lst,physfad_capacity_lst,
                            zogd_time_lst,zogd_capacity,
                            random_time_lst,random_capacity,device):
    # while (loss.item() > -10000 and iters < num_of_iterations):
    plots_lst = []
    legend_lst = []
    if time_lst is not None:


        delta_time_list = [(t-time_lst[0])/ datetime.timedelta(seconds=1) for t in time_lst]
        delta_time_list[0] = delta_time_list[0] + 0.001
        best_capacity = 0
        number_of_consecutive_non_increases = 0
        best_iteration = 0
        for i,capacity in enumerate(physfad_model_capacity_lst):
            number_of_consecutive_non_increases = number_of_consecutive_non_increases + 1
            if capacity>best_capacity:
                best_capacity = capacity
                number_of_consecutive_non_increases = 0
            best_iteration = i
            if number_of_consecutive_non_increases > 40:
                best_iteration = i-40
                break
        p1, = plt.plot(delta_time_list[:best_iteration],model_capacity_lst[:best_iteration])
        p2, = plt.plot(delta_time_list[:best_iteration],physfad_model_capacity_lst[:best_iteration])
        p1.set_label("DNN model")
        p2.set_label("DNN model (tested on physfad)")
        plots_lst.append(p1)
        plots_lst.append(p2)
        legend_lst.append("DNN model")
        legend_lst.append("DNN model (tested on physfad)")
    if physfad_time_lst is not None:
        physfad_delta_time_list = [(t-physfad_time_lst[0])/ datetime.timedelta(seconds=1) for t in physfad_time_lst]
        physfad_delta_time_list[0] = physfad_delta_time_list[0]+0.001
        p3, = plt.plot(physfad_delta_time_list, physfad_capacity_lst)
        p3.set_label("Physfad optimization")
        plots_lst.append(p3)
        legend_lst.append("Physfad optimization")
    if zogd_time_lst is not None:
        zogd_delta_time_list = [(t - zogd_time_lst[0]) / datetime.timedelta(seconds=1) for t in zogd_time_lst]
        zogd_delta_time_list[0] = zogd_delta_time_list[0] + 0.001
        p4, = plt.plot(zogd_delta_time_list, zogd_capacity)
        p4.set_label("Zero Order Gradient Descent")
        plots_lst.append(p4)
        legend_lst.append("Zero Order Gradient Descent")
    if random_time_lst is not None:
        random_delta_time_list = [(t - random_time_lst[0]) / datetime.timedelta(seconds=1) for t in random_time_lst]
        random_delta_time_list[0] = random_delta_time_list[0] + 0.001
        random_capacity = [max(random_capacity[:i+1]) for i,x in enumerate(random_capacity)]
        p5, = plt.plot(random_delta_time_list, random_capacity)
        p5.set_label("Random Search")
        plots_lst.append(p5)
        legend_lst.append("Random Search")
    plt.legend(plots_lst,legend_lst)
    plt.xlabel("seconds[s]")
    plt.ylabel("Channel Capacity")
    plt.xscale('log')
    plt.title("optimization of channel capacity")

    plt.show()
def plus_1_cyclic(counter,max_value,first_loop=True):
    counter = counter+1
    if counter >= max_value:
        counter = 0
        first_loop = False
    return counter, first_loop

def test_model(test_ldr,model,parameters,W,output_size,output_shape,model_output_capacity,device):
    count = 0
    if test_ldr is not None:
        test_NMSE = 0
        for (test_batch_idx, test_batch) in enumerate(test_ldr):
            (X_test, Y_test_Capacity, Y_test) = test_batch
            test_output = model(X_test)
            if model_output_capacity:
                test_NMSE += NMSE(test_output, Y_test_Capacity)
            else:
                test_NMSE += NMSE(test_output, Y_test)
            count = count + 1
        test_NMSE = test_NMSE / count
        if model_output_capacity:
            model_capacity = test_output[0, :]
            physfad_capacity,_ = test_configurations_capacity(parameters,fill_ris_config(X_test[0,:],device),W,device)
        else:
            model_capacity = capacity_loss(test_output[0, :].reshape(1,output_size,output_shape[0],output_shape[1]), torch.ones(output_size,device=device), 1)
            physfad_capacity,_ = test_configurations_capacity(parameters,fill_ris_config(X_test[0,:],device),W,device)
            # model_capacity = -np.inf
            # physfad_capacity = -np.inf
    else:
        test_NMSE = -np.inf
        model_capacity = -np.inf
    return test_NMSE,model_capacity,physfad_capacity

def NMSE(estimations,ground_truth):
    estimations = torch.abs(estimations)
    estimations_c = estimations.cpu().detach().numpy()
    ground_truth_c = ground_truth.cpu().detach().numpy()
    return 100 * (np.mean((estimations_c - ground_truth_c) ** 2) /
                  np.mean(ground_truth_c**2))

def generate_m_random_points_on_Nsphere(m,N,device):
    random_mat = np.random.random((m,N))*2 - 1
    norm_mat = np.expand_dims(np.linalg.norm(random_mat,axis=1),axis=1)
    tensor_output = torch.tensor(random_mat / norm_mat,device=device)
    return tensor_output
def estimate_gradient(func,x,epsilon,m,device):
    N = x.shape[-1]
    rand_vec = generate_m_random_points_on_Nsphere(m,N,device)
    f_x_plus_eps = torch.zeros((m,1), device=device)
    f_x_minus_eps = torch.zeros((m,1), device=device)
    for i,point in enumerate(rand_vec):
        f_x_plus_eps[i] = func(x+epsilon*point)
        f_x_minus_eps[i] = func(x-epsilon*point)


    return torch.sum((f_x_plus_eps-f_x_minus_eps)*rand_vec/(2*epsilon),dim=0)/m

