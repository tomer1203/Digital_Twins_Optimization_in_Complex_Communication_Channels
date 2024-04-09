import numpy as np
import torch
import torch
from matplotlib import pyplot as plt
import scipy.io
import time
import random
from models import *
from dataset import *
from ChannelMatrixEvaluation import (capacity_loss,get_configuration_parameters,test_configurations_capacity,
                                     fill_ris_config, physfad_model, physfad_channel_optimization,
                                     zeroth_grad_optimization,random_search_optimization)
from PhysFadPy import get_bessel_w
import datetime
from functools import reduce
import cProfile,pstats,io
from utils import (print_train_iteration, plot_train_epoch, plot_model_optimization, plus_1_cyclic, test_model,
                   NMSE,test_dnn_optimization)
# -----------------------------------------------------------


MSE_Loss_torch1 = nn.MSELoss()
MSE_Loss_torch2 = nn.MSELoss()


# -----------------------------------------------------------
def My_MSE_Loss(X,Y,Y_capacity,batch_size,output_size,output_shape,model_output_capacity,calc_capacity,device,capacity_maximization_on=False):
    # min_std = torch.min(torch.std(X,0),5*torch.std(Y,0))
    epsilon = 0.00000001

    if model_output_capacity:
        Y=Y_capacity.unsqueeze(1)
    # else:
        # X_Norm = (torch.abs(X) / torch.norm(torch.abs(X), p=1, dim=1).unsqueeze(1))+epsilon
        # Y_Norm = (torch.abs(Y) / torch.norm(torch.abs(Y), p=1, dim=1).unsqueeze(1))+epsilon
        # KL_divergence = torch.sum(X_Norm * torch.log(X_Norm / Y_Norm))
        # mean_MSE = torch.sum((torch.norm(torch.abs(X), p=1, dim=1).unsqueeze(1) - torch.norm(torch.abs(Y), p=1,
        #                                                                                      dim=1).unsqueeze(1)) ** 2)

    if X.shape[0]==1:
        min_std = torch.Tensor(0)
    else:
        # min_std = MSE_Loss_torch2(torch.std(X,0),torch.std(Y,0))
        min_std = (torch.std(X, 0) - torch.std(Y, 0)) ** 2

    MSE_Loss = torch.sum((X-Y)**2)
    # MSE_Loss = MSE_Loss_torch1(X,Y)
    if calc_capacity:
        X_capacity = capacity_loss(X.reshape(batch_size,output_size,output_shape[0],output_shape[1]),torch.ones(output_size,device=device),1,list_out=True)
        capacity_mse = (X_capacity-Y_capacity)**2
    # Y_capacity = capacity_loss(Y.reshape(batch_size,output_size,output_shape[0],output_shape[1]),torch.ones(output_size,device=device),1)
    # min_std = torch.where(torch.std(X,0)>torch.std(Y,0),torch.std(Y,0),0)
    # print(min_std==0)
    # sum_MSE = (torch.sum(X)-torch.sum(Y))**2
    # if torch.sum(torch.std(X,0)) > torch.sum(torch.std(Y,0)):
        # print("Y",end="")
    # else:
        # print(str(torch.sum(torch.std(X,0))/torch.sum(torch.std(Y,0))),end=" ")

    # print("MSE ", end="")
    # print(MSE_Loss)
    # print("kl ", end="")
    # print(100*KL_divergence)
    # print("mean ", end="")
    # print(mean_MSE / 100)
    # print("variance ", end="")
    # print(torch.sum(min_std)/5)


    # loss = KL_divergence
    loss = MSE_Loss
    # loss = loss + 30*KL_divergence
    loss = loss + 100*torch.sum(min_std)# 100
    # loss = loss + 5*min_std# 100
    if calc_capacity:
        loss = loss + torch.sum(capacity_mse)/2
    if capacity_maximization_on:
        loss = loss - torch.sum(X_capacity)/3
        # print("capacity ", end="")
        # print(torch.mean(capacity_mse)/10)
    # loss = loss + mean_MSE/100
    # loss = MSE_Loss+50*KL_divergence+100*torch.sum(min_std)#+mean_MSE/100
    # loss = 100*KL_divergence+mean_MSE/100#+120*torch.sum(capacity_mse)
    # if torch.any(~torch.isfinite(loss)):
    #     print("non finite value detected")
    return loss

def optimize_n_iterations(X,n,net,W,parameters,batch_size,output_size,output_shape,lr,cut_off=-np.inf,model_output_capacity=False,device="cpu"):
    loss = torch.Tensor([1])
    optimizer = torch.optim.Adam([X],lr=lr)#weight_decay = 0.00001# weight_decay = 0.001
    iters = 0
    net.eval()
    # orig_requires_grads = [w.requires_grad for w in net.parameters()]
    # for w in net.parameters():
        # print("turning weight grad off")
        # w.requires_grad = False
    P = torch.ones(output_size,device=device)
    while (loss.item()>cut_off and iters < n):
        optimizer.zero_grad()
        pred = net(X)
        if model_output_capacity:
            loss = -torch.sum(pred)
            loss = loss/batch_size
        else:
            # try:
            loss = -capacity_loss(pred.reshape(batch_size,output_size,output_shape[0],output_shape[1]),P,1)
            # except Exception as inst:
            #     print("capacity calculation thrown an error " + str(inst.args))
            #     return X,-loss.item()
        # print("optimizing... {0}  {1}".format(iters,-loss.item()))
        # loss_with_reg = loss+torch.sum(torch.abs(X))
        loss.backward()
        optimizer.step()
        iters = iters + 1
    # for w, rg in zip(net.parameters(), orig_requires_grads): w.requires_grad = rg
    return X,-loss.item()
def train_and_optimize(model,
                       loss_func, optimizer,
                       train_ds, test_ldr,
                       max_epochs, epoch_cut_off, ep_log_interval,
                       batch_size, output_size, output_shape,
                       model_output_capacity,
                       test_optim_chance,initial_optim_steps,step_increase, optim_lr,
                       NMSE_LST_SIZE,load_new_data,RIS_config_file=None, H_realiz_file=None, capacity_file=None,device='cpu'):
    NMSE_lst = np.zeros(NMSE_LST_SIZE)
    time_list_size = 200
    time_list_idx = 0
    first_time_loop = True
    iteration_time_list = [datetime.datetime.now()-datetime.datetime.now()]*time_list_size
    NMSE_idx = 0
    optim_step = initial_optim_steps
    parameters = get_configuration_parameters(device)
    optimized_inputs = None
    optimized_outputs = None
    optimized_capacity = None
    current_time = datetime.datetime.now()
    (freq, x_tx, y_tx, _, _, _,x_rx, y_rx, _, _, _,x_env, y_env, _, _, _, x_ris_c, y_ris_c) = parameters
    W = get_bessel_w(freq,x_tx, y_tx,x_rx, y_rx,x_env, y_env,x_ris_c, y_ris_c, device)
    model.train()
    if load_new_data:
        train_ds = RISDataset(RIS_config_file, H_realiz_file, capacity_file, calculate_capacity=False,
                              output_size=output_size, output_shape=output_shape, device=device)

    train_ldr = T.utils.data.DataLoader(train_ds,batch_size=batch_size, shuffle=True)
    for epoch in range(0, max_epochs):
        T.manual_seed(1 + epoch)

        print("epoch {0} train dataset length {1}".format(epoch,batch_size*len(train_ldr)))
        for (batch_idx, batch) in enumerate(train_ldr):
            last_time = current_time
            current_time = datetime.datetime.now()
            iteration_time_list[time_list_idx] = current_time-last_time

            (X, Y_capacity, Y) = batch  # (predictors, targets)
            for param in model.parameters():  # fancy zero_grad
                param.grad = None
            oupt = model(X)
            loss_val = loss_func(oupt, Y, Y_capacity, oupt.shape[0], output_size, output_shape,model_output_capacity,calc_capacity=False,device=device)  # avg per item in batch
            loss_val.backward()  # compute gradients
            optimizer.step()  # update wts
            if model_output_capacity:
                NMSE_lst[NMSE_idx] = NMSE(oupt, Y_capacity.unsqueeze(1))
            else:
                NMSE_lst[NMSE_idx] = NMSE(oupt, Y)
            NMSE_idx,_ = plus_1_cyclic(NMSE_idx,NMSE_LST_SIZE)
            time_list_idx,first_time_loop = plus_1_cyclic(time_list_idx,time_list_size,first_time_loop)


            # print("progress: "+str(100*(batch_idx/len(train_ldr))).split(".")[0]+"%"+" Train NMSE: {0:.4f}".format(np.mean(NMSE_lst)))
            # if batch_idx/len(train_ldr) > epoch_cut_off:
            if batch_idx > 699:
                break

            if random.random() < test_optim_chance:

                # optimize for optim_steps steps
                a_t = datetime.datetime.now()
                X_copy = X.clone().detach().requires_grad_(True).to(device)
                number_of_optimization_steps = random.randint(1,optim_step)
                X_opt,model_capacity = optimize_n_iterations(X_copy,
                                                             number_of_optimization_steps,
                                                             model, W, parameters,
                                                             batch_size, output_size, output_shape,
                                                             lr=optim_lr,
                                                             model_output_capacity=model_output_capacity,
                                                             device=device)

                X_opt = X_opt.clone().detach().to(device)
                # try:
                with torch.no_grad():
                    Y_opt_capacity,Y_opt_gt = test_configurations_capacity(parameters, fill_ris_config(X_opt ** 2, device), W, device,list_out=True)
                # except Exception as inst:
                #     print("capacity calculation thrown an error " + str(inst.args))
                #     print(X_opt)
                #     print(X_opt**2)
                #     print(torch.any(~torch.isfinite(X_opt**2)))
                # for idx in range(batch_size):
                #     Y_idx = physfad_model(parameters,X_opt[idx,:].unsqueeze(0),W,device)
                #     if torch.equal(Y_idx,Y_opt_gt[idx]):
                #         print(" True")
                #     else:
                #         print(" False")
                Y_opt_gt_flattended = Y_opt_gt.reshape(batch_size,-1)

                # calculate loss
                model.train()
                optimizer.zero_grad()
                Y_model_opt = model(X_opt)
                loss_val = loss_func(Y_model_opt, torch.abs(Y_opt_gt_flattended), Y_opt_capacity, Y_model_opt.shape[0],
                                     output_size,output_shape,
                                     model_output_capacity,calc_capacity=True,device=device,capacity_maximization_on=True) # avg per item in batch
                # backpropegate
                loss_val.backward()
                optimizer.step()  # update wts
                if model_output_capacity:
                    optimized_NMSE = NMSE(Y_model_opt, Y_opt_capacity.unsqueeze(1))
                else:
                    optimized_NMSE = NMSE(Y_model_opt, torch.abs(Y_opt_gt_flattended))
                # time_elapsed = datetime.datetime.now() - start_time
                time_per_iteration = reduce(lambda x, y: x + y, iteration_time_list) / (time_list_idx if first_time_loop else time_list_size)
                epochs_left = max_epochs-epoch-1
                if test_optim_chance<0 or test_optim_chance>1:
                    geometric_series_factor = -np.inf
                else:
                    growth_rate = 1+epoch_cut_off*test_optim_chance
                    geometric_series_factor = ((1-(growth_rate)**epochs_left)/(1-growth_rate)) - 1
                length_of_current_epoch = len(train_ldr)*epoch_cut_off
                time_left = time_per_iteration * (length_of_current_epoch*geometric_series_factor + length_of_current_epoch - batch_idx - 1)
                time_left2 = time_per_iteration*(700*epochs_left+(700-batch_idx))
                # print(str(time_left2).split(".")[0]+"s",end = " ")
                print("E{0} I{1:.0f}% {2} steps: {3} train NMSE: {4:.4f} optimized train NMSE: {5:.4f} model capacity {6:.4f} physfad capacity {7:.4f} {8:.4f} sum(X): {9:.0f}  {10:.0f}".format(
                    str(epoch),
                    # 100 * batch_idx / (epoch_cut_off*len(train_ldr)),
                    100 * (batch_idx) / 700,
                    str(time_left2).split(".")[0]+"s",
                    number_of_optimization_steps,
                    np.mean(NMSE_lst),
                    optimized_NMSE,
                    model_capacity,
                    Y_capacity.mean().cpu().detach().numpy(),
                    Y_opt_capacity.mean().cpu().detach().numpy(),
                    torch.sum(X_opt).cpu().detach().numpy()/ batch_size,
                    torch.sum(X).cpu().detach().numpy()/ batch_size))
                # add optimized_input,Y_opt to next epochs dataset.
                if torch.any(~torch.isfinite(X_opt)):
                    print("non finite value detected")
                if optimized_inputs is None:
                    optimized_inputs = fill_ris_config(X_opt,device)
                    optimized_outputs = torch.abs(Y_opt_gt_flattended)
                    optimized_capacity = Y_opt_capacity
                else:
                    optimized_inputs = torch.vstack([optimized_inputs,fill_ris_config(X_opt,device)])
                    optimized_outputs = torch.vstack([optimized_outputs,torch.abs(Y_opt_gt_flattended)])
                    optimized_capacity = torch.hstack([optimized_capacity,Y_opt_capacity])
        # add to dataset
        train_ds.add_new_items(optimized_inputs,optimized_outputs,optimized_capacity)
        optimized_inputs,optimized_outputs,optimized_capacity = None,None,None
        if train_ds.dataset_changed:
            train_ds.save_dataset(RIS_config_file, H_realiz_file, capacity_file)
            train_ldr = T.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        optim_step = optim_step+step_increase

def train(model, loss_func, optimizer, train_ldr, test_ldr, max_epochs, ep_log_interval, output_size, output_shape, model_output_capacity, NMSE_LST_SIZE, device):
    NMSE_lst = np.zeros(NMSE_LST_SIZE)
    variation_list = np.zeros(NMSE_LST_SIZE)
    capacity_list = np.zeros(NMSE_LST_SIZE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=3)
    NMSE_idx = 0
    NMSE_Train = []
    NMSE_TEST = []
    model.train()  # set mode
    parameters = get_configuration_parameters(device)
    # pr = cProfile.Profile()
    # pr.enable()
    (freq, x_tx, y_tx, _, _, _, x_rx, y_rx, _, _, _, x_env, y_env, _, _, _, x_ris_c, y_ris_c) = parameters
    W = get_bessel_w(freq, x_tx, y_tx, x_rx, y_rx, x_env, y_env, x_ris_c, y_ris_c, device)
    for epoch in range(0, max_epochs):
        T.manual_seed(1 + epoch)  # recovery reproducibility
        epoch_loss = 0  # for one full epoch
        start_time = datetime.datetime.now()
        for (batch_idx, batch) in enumerate(train_ldr):
            (X,Y_capacity, Y) = batch  # (predictors, targets)
            # optimizer.zero_grad()  # prepare gradients
            for param in model.parameters(): # fancy zero_grad
                param.grad = None
            oupt = model(X)
            # print(torch.sum(oupt).item())
            loss_val = loss_func(oupt, Y,Y_capacity,oupt.shape[0], output_size,output_shape,model_output_capacity,calc_capacity=True,device=device)  # avg per item in batch


            epoch_loss += loss_val.item()  # accumulate avgs
            loss_val.backward()  # compute gradients
            optimizer.step()  # update wts
            if model_output_capacity:
                NMSE_lst[NMSE_idx] = NMSE(oupt, Y_capacity.unsqueeze(1))
                variation_list[NMSE_idx] = (torch.mean(torch.std(oupt, 0) / torch.std(Y_capacity, 0))).item()
            else:
                NMSE_lst[NMSE_idx] = NMSE(oupt, Y)
                variation_list[NMSE_idx] = (torch.mean(torch.std(oupt, 0) / torch.std(Y, 0))).item()
                estimated_capacity = capacity_loss(oupt.reshape(oupt.shape[0],output_size,output_shape[0],output_shape[1]),torch.ones(output_size,device=device),1,list_out=True)
                capacity_list[NMSE_idx] = NMSE(estimated_capacity,Y_capacity)
            ## DEBUG
            # physfad_capacity, physfad_H = test_configurations_capacity(parameters, fill_ris_config(X[0,:], device),
            #                                                            device)
            # optimization_nmse = NMSE(oupt[0,:], torch.abs(physfad_H).reshape(1, -1))
            # print("Matlab: {0}".format(NMSE_lst[NMSE_idx]))
            # print("Python: {0}\n\n".format(optimization_nmse))
            ## END DEBUG


            if False and batch_idx%300==0:
                print_train_iteration(start_time, batch_idx, epoch, max_epochs, train_ldr, pr=None,NMSE_lst=NMSE_lst)

            NMSE_idx,_ = plus_1_cyclic(NMSE_idx, NMSE_LST_SIZE)


        if epoch % ep_log_interval == 0:
            if epoch % 3*ep_log_interval ==0:
                if not model_output_capacity:
                    plot_train_epoch(Y,oupt,model_output_capacity)
            test_NMSE,model_capacity,physfad_capacity = test_model(test_ldr,model,parameters,W,output_size,output_shape,model_output_capacity,device)

            scheduler.step(test_NMSE)
            # print("NMSE for test is "+str(test_NMSE))
            time_elapsed = datetime.datetime.now() - start_time
            time_per_epoch = time_elapsed / (epoch + 1)
            completed = epoch / max_epochs
            time_left = time_per_epoch  * (max_epochs - epoch - 1)
            print("epoch =%4d time left =%s  loss =%0.4f NMSE =%0.5f test_NMSE =%0.5f output normalized variation=%0.5f, training capacity NMSE=%0.5f model capacity %0.3f, phsyfad capacity %0.3f" % \
                  (epoch,str(time_left).split(".")[0], epoch_loss, np.mean(NMSE_lst), test_NMSE, np.mean(variation_list), np.mean(capacity_list), model_capacity,physfad_capacity))
            NMSE_Train.append(np.mean(NMSE_lst))
            NMSE_TEST.append(test_NMSE)
            # save checkpoint
            # dt = time.strftime("%Y_%m_%d-%H_%M_%S")
            # fn = ".\\Log\\" + str(dt) + str("-") + \
            #      str(epoch) + "_checkpoint.pt"

            # info_dict = {
            #     'epoch': epoch,
            #     'net_state': net.state_dict(),
            #     'optimizer_state': optimizer.state_dict()
            # }
            # T.save(info_dict, fn)
    return (NMSE_Train, NMSE_TEST)
def load_data(batch_size,output_size,output_shape,device):
    train_RIS_file = "..\\Data\\full_range_RISConfiguration.mat" # full_range_RISConfiguration
    # train_RIS_file = "..\\Data\\new_full_RISConfiguration.mat"
    train_H_file = "..\\Data\\full_range_H_realizations.mat" # full_range_H_realizations
    # train_H_file = "..\\Data\\new_full_H_realizations.mat"
    train_H_capacity_file = "..\\Data\\full_H_capacity.txt" # full_H_capacity
    train_ds = RISDataset(train_RIS_file, train_H_file, train_H_capacity_file, calculate_capacity=False,
                          output_size=output_size, output_shape=output_shape, device=device)
    test_RIS_file = "..\\Data\\Test_full_RISConfiguration.mat"
    test_H_file = "..\\Data\\Test_full_H_realizations.mat"
    test_H_capacity_file = "..\\Data\\Test_full_H_capacity.txt"
    test_ds = RISDataset(test_RIS_file, test_H_file, test_H_capacity_file, calculate_capacity=False,
                         output_size=output_size, output_shape=output_shape, device=device)

    train_ldr = T.utils.data.DataLoader(train_ds,
                                        batch_size=batch_size, shuffle=True)
    test_ldr = T.utils.data.DataLoader(test_ds,
                                       batch_size=batch_size, shuffle=True)
    return train_ds,test_ds,train_ldr,test_ldr
def optimize(starting_inp,net,inp_size,output_size,optim_lr = 0.005, model_output_capacity=False,
             calaculate_physfad=True,device='cpu',num_of_iterations=1500,noise=1):
    net.train()
    # run gradiant descent to find best input configuration
    batch_size = 1
    # (X, _, _) = next(iter(train_ldr))  # (predictors, targets)
    # estOptInp = torch.randn([batch_size, inp_size], device=device)
    # estOptInp[estOptInp >= 0] = 5 - 2 * (estOptInp[estOptInp >= 0])
    # estOptInp[estOptInp < 0] = 1 + (0.15 + estOptInp[estOptInp < 0]) * 0.3
    # estOptInp = starting_inp[0, :].unsqueeze(0).clone().detach().requires_grad_(True).to(device)  # Activate gradients
    estOptInp = starting_inp
    time_lst = []
    opt_inp_lst = []
    model_capacity_lst = []
    # estOptInp = torch.tensor([batch_size,inp_size],requires_grad=True,device=device).uniform_(0.9,5)
    # estOptInp = torch.from_numpy(np.random.uniform(low=0.9, high=5, size=(batch_size, inp_size)),device=device,requires_grad=True)
    # print(estOptInp)
    # print(X)
    Inp_optimizer = torch.optim.Adam([estOptInp], lr=optim_lr)# weight_decay = 0.00001


    loss = torch.Tensor([1])
    # num_of_iterations = 1500
    loss_list_size = 1
    iters = -loss_list_size
    loss_list = np.zeros(loss_list_size)
    parameters = get_configuration_parameters(device)
    # while (loss.item() > -10000 and iters < num_of_iterations):
    (freq, x_tx, y_tx, _, _, _, x_rx, y_rx, _, _, _, x_env, y_env, _, _, _, x_ris_c, y_ris_c) = parameters
    W = get_bessel_w(freq, x_tx, y_tx, x_rx, y_rx, x_env, y_env, x_ris_c, y_ris_c, device)
    net.eval()
    pred = net(estOptInp)
    if model_output_capacity:
        loss = -torch.sum(pred)
        loss = loss / batch_size
    else:
        loss = -capacity_loss(pred.reshape(batch_size, output_size, 4, 3), torch.ones(output_size, device=device), noise)
        # loss = -torch.sum(pred)

    time_lst.append(datetime.datetime.now())
    opt_inp_lst.append(estOptInp.clone())
    model_capacity_lst.append(-loss.item())
    while (iters < num_of_iterations):
        Inp_optimizer.zero_grad()
        pred = net(estOptInp)
        if model_output_capacity:
            loss = -torch.sum(pred)
            loss = loss/batch_size
        else:
            loss = -capacity_loss(pred.reshape(batch_size, output_size, 4, 3), torch.ones(output_size,device=device), noise)
            # loss = -torch.sum(pred)
        # loss_with_reg = loss+torch.sum(torch.abs(estOptInp))/1000
        # loss = -torch.sum(pred)
        loss.backward()
        Inp_optimizer.step()

        loss_list[iters % loss_list_size] = -loss.item()
        if iters % 20 == 0:
            if calaculate_physfad:
                physfad_capacity, physfad_H = test_configurations_capacity(parameters,
                                                                           fill_ris_config(estOptInp ** 2, device), W,
                                                                           device,noise=noise)
                if model_output_capacity:
                    optimization_nmse = -np.inf
                else:
                    optimization_nmse = NMSE(pred.reshape(120, 4, 3), torch.abs(physfad_H))
            else:
                physfad_capacity = -np.inf
                optimization_nmse = -np.inf
            print(
                "iter #{0} input distance from zero: {1:.2f} model capacity loss: {2:.4f} physfad capacity {3:.4f}, NMSE {4:.4f}".format(
                    iters, torch.abs(
                        estOptInp).sum().cpu().item() / batch_size, loss_list.mean() / batch_size, physfad_capacity,
                    optimization_nmse))
            time_lst.append(datetime.datetime.now())
            opt_inp_lst.append(estOptInp.clone())
            model_capacity_lst.append(-loss.item())
            # print(
            #     "iter #{0} input distance from zero: {1:.2f} model capacity loss: {2:.4f} physfad capacity {3:.4f}".format(
            #         iters, torch.abs(estOptInp).sum().cpu().item() / batch_size, loss_list.mean() / batch_size, physfad_capacity))
            # plt.plot(torch.mean(test_pred,dim=0).t().cpu().detach().numpy())
            # plt.legend(["asa,asd"])
            # plt.show()
        iters = iters + 1
    print("Done optimization")
    print(estOptInp)
    np.savetxt("optimal_parameters.txt", estOptInp[0, :].cpu().detach().numpy())
    return (time_lst,opt_inp_lst,model_capacity_lst)
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu" # force choosing cpu
    print(device)
    # 0. get started
    # T.manual_seed(4)  # representative results
    # np.random.seed(4)

    batch_size = 32
    output_size = 120
    output_shape = (4, 3)
    inp_size = 264  # 264
    model_output_capacity = False
    # 1. create DataLoader objects

        # 2. create network
    # net = Net(inp_size,50,120,(4,3)).to(device)
    net = Net(inp_size,50,120,(4,3),model_output_capacity).to(device)

    # 3. train model
    max_epochs = 50
    ep_log_interval = 1
    lrn_rate = 0.00005
    optim_lr = 0.008
    num_of_opt_iter = 3000
    load_model = True
    training_mode = False
    activate_train_and_optimize = False  # can be added to the train mode
    optimize_model = True
    run_snr_graph = False
    loss_func = My_MSE_Loss
    # optimizer = T.optim.SGD(net.parameters(), lr=lrn_rate)
    optimizer = T.optim.Adam(net.parameters(), lr=lrn_rate)  # weight_decay=0.001

    print("\nbatch_size = %3d " % batch_size)
    print("loss = " + str(loss_func))
    print("optimizer = Adam")
    print("max_epochs = %3d " % max_epochs)
    print("train learning rate = %0.4f " % lrn_rate)
    print("optimization learning rate = %f " % optim_lr)


    NMSE_LST_SIZE = 10
    print("Collecting RIS configuration ")
    train_ds,test_ds,train_ldr, test_ldr = load_data(batch_size,output_size,output_shape,device)
    if load_model:
        print("Loading model")
        net.load_state_dict(torch.load(".\\Models\\large_model_long_tr_op_loop.pt"))
        # net.load_state_dict(torch.load(".\\Models\\Full_Main_model.pt"))
        optimizer = T.optim.Adam(net.parameters(), lr=lrn_rate)
    if training_mode:
        print("\nStarting training with saved checkpoints")
        (NMSE_Train, NMSE_TEST) = train(net,
                                        loss_func,
                                        optimizer,
                                        train_ldr,
                                        test_ldr,
                                        max_epochs,
                                        ep_log_interval,
                                        output_size,
                                        output_shape,
                                        model_output_capacity,
                                        NMSE_LST_SIZE,
                                        device=device)
        print("Done Training")
        plt.plot(range(0,max_epochs,ep_log_interval),NMSE_Train, "r")
        plt.plot(range(0,max_epochs,ep_log_interval),NMSE_TEST, "b")
        plt.legend(["Train NMSE", "Test NMSE"])
        plt.xlabel("epochs")
        plt.show()
        torch.save(net.state_dict(),".\\Models\\Full_Main_model.pt")

    # if device != torch.device('cpu'):
    #     print("Moving data to cpu")
    #     device = torch.device('cpu')
    #     net = net.to(device)
    #     train_ds, test_ds, train_ldr, test_ldr = load_data(batch_size, output_size, output_shape, device)
    #     optimizer = T.optim.Adam(net.parameters(), lr=lrn_rate)


    if activate_train_and_optimize:
        print("Starting deep-training loop with optimization")
        batch_size = 32
        max_epochs = 120
        initial_optim_steps = 10
        step_increase = 10
        test_optim_chance = 0.015
        epoch_cut_off = 0.05
        load_new_data = False

        print("batch_size: " + str(batch_size))
        print("max_epochs: " + str(max_epochs))
        print("initial_optim_steps: " + str(initial_optim_steps))
        print("step_increase: " + str(step_increase))
        print("test_optim_chance: " + str(test_optim_chance))
        print("epoch_cut_off: " + str(epoch_cut_off))
        print("load_new_data: " + str(load_new_data))


        optimized_train_RIS_file = "..\\Data\\optimized_RISConfiguration.mat"
        optimized_train_H_file = "..\\Data\\optimized_H_realizations.mat"
        optimized_train_H_capacity_file = "..\\Data\\optimized_H_capacity.txt"
        train_and_optimize(net,
                           loss_func,
                           optimizer,
                           train_ds,test_ldr,
                           max_epochs, epoch_cut_off, ep_log_interval,
                           batch_size, output_size, output_shape,
                           model_output_capacity,
                           test_optim_chance, initial_optim_steps, step_increase,optim_lr,
                           NMSE_LST_SIZE,
                           load_new_data, optimized_train_RIS_file, optimized_train_H_file, optimized_train_H_capacity_file,device)

        torch.save(net.state_dict(), ".\\Models\\Full_Main_model.pt")
    if optimize_model:
        device_cpu = torch.device('cpu')
        device_cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # optimized_train_RIS_file = "..\\Data\\large_model_optimized_RISConfiguration.mat"
        # optimized_train_H_file = "..\\Data\\large_model_optimized_H_realizations.mat"
        # optimized_train_H_capacity_file = "..\\Data\\large_model_optimized_H_capacity.txt"
        # train_ds = RISDataset(optimized_train_RIS_file, optimized_train_H_file, optimized_train_H_capacity_file, calculate_capacity=False,
        #                       output_size=output_size, output_shape=output_shape, device=device)
        #
        # train_ldr = T.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        (X, _, _) = next(iter(train_ldr))  # (predictors, targets)
        initial_inp = X[0, :].unsqueeze(0).clone().detach().requires_grad_(True).to(device)
        print("Running Optimization on DL Model")
        physfad_time_lst, physfad_capacity = None,None
        zogd_time_lst, zogd_capacity = None,None
        # (physfad_time_lst,physfad_capacity,physfad_inputs) = physfad_channel_optimization(device,initial_inp.clone().detach().requires_grad_(True).to(device))
        # (zogd_time_lst,zogd_capacity)                      = zeroth_grad_optimization(device,initial_inp.clone().detach().to(device))
        # (time_lst,opt_inp_lst,model_capacity_lst)          = optimize(initial_inp.clone().detach().requires_grad_(True),net,inp_size,output_size,optim_lr,model_output_capacity,num_of_iterations=num_of_opt_iter,calaculate_physfad=False,device=device)
        (X, _, _) = next(iter(train_ldr))  # (predictors, targets)
        initial_inp = X[0, :].unsqueeze(0).clone().detach().requires_grad_(True).to(device_cpu)

        (physfad_time_lst, physfad_capacity, physfad_inputs) = physfad_channel_optimization(device_cpu,
                                                                                            initial_inp.clone().detach().requires_grad_(
                                                                                                True).to(device_cpu),
                                                                                            noise_power=1,
                                                                                            num_of_iterations=50)
        (zogd_time_lst, zogd_capacity) = zeroth_grad_optimization(device_cpu,
                                                                  initial_inp.clone().detach().to(device_cpu),
                                                                  noise_power=1, num_of_iterations=50)
        (rand_search_time_lst, random_search_capacity) = random_search_optimization(50, device_cpu, noise_power=1,
                                                                                    initial_inp=initial_inp.clone().detach().to(device_cpu))

        (time_lst, opt_inp_lst, model_capacity_lst) = optimize(
            initial_inp.to(device_cuda).clone().detach().requires_grad_(True), net, inp_size, output_size, optim_lr,
            model_output_capacity, num_of_iterations=5000, noise=1,#num_of_opt_iter
            calaculate_physfad=False, device=device_cuda)

        dnn_physfad_capacity_lst = test_dnn_optimization([x.cpu() for x in opt_inp_lst], device_cpu, noise=1)

        max_index = dnn_physfad_capacity_lst.argmax()

        time_lst = np.array(time_lst)
        physfad_time_lst = np.array(physfad_time_lst)
        zogd_time_lst = np.array(zogd_time_lst)
        rand_search_time_lst = np.array(rand_search_time_lst)

        physfad_capacity = np.array(physfad_capacity)
        zogd_capacity = np.array([x.detach().numpy() for x in zogd_capacity])
        random_search_capacity = np.array([x.detach().numpy() for x in random_search_capacity])
        np.save(".\\outputs\\time_lst.npy",time_lst)
        np.save(".\\outputs\\physfad_time_lst.npy",physfad_time_lst)
        np.save(".\\outputs\\zogd_time_lst.npy",zogd_time_lst)
        np.save(".\\outputs\\rand_search_time_lst.npy",rand_search_time_lst)
        np.save(".\\outputs\\dnn_physfad_capacity_lst.npy",dnn_physfad_capacity_lst)
        np.save(".\\outputs\\physfad_capacity.npy",physfad_capacity)
        np.save(".\\outputs\\zogd_capacity.npy",zogd_capacity)
        np.save(".\\outputs\\random_search_capacity.npy",random_search_capacity)

        plot_model_optimization(time_lst,dnn_physfad_capacity_lst,model_capacity_lst,physfad_time_lst,physfad_capacity,zogd_time_lst,zogd_capacity,rand_search_time_lst,random_search_capacity,device)


    if run_snr_graph:
        device_cpu = torch.device('cpu')
        device_cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_snr_points = 15
        attempts = 6
        snr_values = np.linspace(0.5,60,num_snr_points)

        noise_array = 1/snr_values
        SNR_results = np.zeros([num_snr_points,7,attempts])
        for i,noise in enumerate(noise_array):
            for attempt in range(attempts):
                print(noise,i,attempt)
                (X, _, _) = next(iter(train_ldr))  # (predictors, targets)
                initial_inp = X[0, :].unsqueeze(0).clone().detach().requires_grad_(True).to(device_cpu)

                (physfad_time_lst, physfad_capacity, physfad_inputs) = physfad_channel_optimization(device_cpu,
                                                                                                    initial_inp.clone().detach().requires_grad_(True).to(device_cpu),
                                                                                                    noise_power=noise,num_of_iterations=150)
                (zogd_time_lst, zogd_capacity) = zeroth_grad_optimization(device_cpu, initial_inp.clone().detach().to(device_cpu),
                                                                          noise_power=noise,num_of_iterations=200)
                (rand_search_time_lst, random_search_capacity) = random_search_optimization(300,device_cpu,noise_power=noise)

                (time_lst, opt_inp_lst, model_capacity_lst) = optimize(
                    initial_inp.to(device_cuda).clone().detach().requires_grad_(True), net, inp_size, output_size,optim_lr,
                    model_output_capacity,num_of_iterations=num_of_opt_iter,noise=noise,#num_of_opt_iter
                    calaculate_physfad=False, device=device_cuda)

                dnn_physfad_capacity_lst=test_dnn_optimization([x.cpu() for x in opt_inp_lst],device_cpu,noise=noise)

                max_index = dnn_physfad_capacity_lst.argmax()


                time_lst                = np.array(time_lst)
                physfad_time_lst        = np.array(physfad_time_lst)
                zogd_time_lst           = np.array(zogd_time_lst)
                rand_search_time_lst    = np.array(rand_search_time_lst)

                physfad_capacity = np.array(physfad_capacity)
                zogd_capacity = np.array([x.detach().numpy() for x in zogd_capacity])
                random_search_capacity = np.array([x.detach().numpy() for x in random_search_capacity])
                SNR_results[i, 0, attempt] = physfad_capacity[-1]
                SNR_results[i, 1, attempt] = zogd_capacity[-1]
                SNR_results[i, 2, attempt] = max(random_search_capacity)
                SNR_results[i, 3, attempt] = dnn_physfad_capacity_lst[max_index]
                SNR_results[i, 4, attempt] = physfad_capacity[physfad_time_lst-physfad_time_lst[0]< time_lst[max_index]-time_lst[0]][-1]
                SNR_results[i, 5, attempt] = zogd_capacity[zogd_time_lst-zogd_time_lst[0]< time_lst[max_index]-time_lst[0]][-1]
                SNR_results[i, 6, attempt] = max(random_search_capacity[rand_search_time_lst - rand_search_time_lst[0] < time_lst[max_index] - time_lst[0]])
        np.save(".\\outputs\\SNR_results.npy",SNR_results)
        plt.plot(1/noise_array,np.mean(SNR_results[:,0,:],axis=1))
        plt.plot(1/noise_array,np.mean(SNR_results[:,1,:],axis=1))
        plt.plot(1/noise_array,np.mean(SNR_results[:,2,:],axis=1))
        plt.plot(1/noise_array,np.mean(SNR_results[:,3,:],axis=1))
        plt.plot(1/noise_array,np.mean(SNR_results[:,4,:],axis=1))
        plt.plot(1/noise_array,np.mean(SNR_results[:,5,:],axis=1))
        plt.plot(1/noise_array,np.mean(SNR_results[:,6,:],axis=1))
        plt.legend(["physfad","zero order gradient descent","random search","dnn","physfad limited","zogd limited","random search limited"])
        # plt.legend(["random search","dnn","random search limited"])
        # plt.legend(["dnn 0.003","dnn 0.005","dnn 0.008","dnn 0.01"])
        plt.show()

    print("\nEnd Simulation")
if __name__ == "__main__":
    enclosure = {}
    scipy.io.loadmat("..//PhysFad//ComplexEnclosure.mat",enclosure)

    main()
    # cProfile.run('main()')

