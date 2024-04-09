import scipy.io
import numpy as np
import torch
import random
from PhysFadPy import GetH,get_bessel_w
import matplotlib.pyplot as plt
import cProfile
import cProfile,pstats,io
import datetime


def physfad_model(parameters,ris_configuration,W,device):
    (freq,x_tx,y_tx,fres_tx,chi_tx,gamma_tx,
         x_rx,y_rx,fres_rx,chi_rx,gamma_rx,
         x_env,y_env,fres_env,chi_env,gamma_env,x_ris_c,y_ris_c) = parameters
    if not torch.is_tensor(ris_configuration):
        ris_configuration = torch.tensor(ris_configuration,device=device)
    fres_ris_c = ris_configuration[:,0:88]
    chi_ris_c = ris_configuration[:,88:176]
    gamma_ris_c = ris_configuration[:,176:264]

    # pr = cProfile.Profile()
    # pr.enable()
    H = GetH(freq,W,
         x_tx, y_tx, fres_tx, chi_tx, gamma_tx,
         x_rx, y_rx, fres_rx, chi_rx, gamma_rx,
         x_env, y_env, fres_env, chi_env, gamma_env,
         x_ris_c, y_ris_c, fres_ris_c, chi_ris_c, gamma_ris_c, device)
    # pr.disable()
    # s = io.StringIO()
    # sortby = 'cumulative'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # pr.dump_stats("train_time_profile.prof")
    # exit()
    return H



def fill_ris_config(fres,device):
    '''
        in case the input is only resonant frequency we need to repad it with all the rest of the ris configuration
    '''
    if len(fres.shape)==1:#single element,no batch
        batch_size = 1
        number_of_elements = fres.shape[0]
        fres = fres.unsqueeze(0)
        if len(fres) == 264:# filling not needed, return original
            return fres
    else:
        batch_size = fres.shape[0]
        number_of_elements = fres.shape[1]
        if fres.shape[1] == 264:# filling not needed, return original
            return fres

    chi_ris = 0.2 * torch.ones((batch_size,number_of_elements),dtype=torch.float64,device=device)
    gamma_ris = 0 * torch.zeros((batch_size,number_of_elements),dtype=torch.float64,device=device)

    return torch.hstack([fres,chi_ris,gamma_ris])
def get_configuration_parameters(device):
    '''
        get the basic ris configuration and pack it into one tuple: parameters
    '''
    freq = torch.tensor(np.linspace(0.9, 1.1, 120));

    ## Configurable Dipole Properties
    ## Transmitters ##
    # locations
    x_tx = torch.tensor([0, 0, 0]).unsqueeze(0).to(device)
    y_tx = torch.tensor([4, 4.5, 5]).unsqueeze(0).to(device)
    # dipole properties
    fres_tx = torch.tensor([1, 1, 1]).unsqueeze(0).to(device)
    chi_tx = torch.tensor([0.5, 0.5, 0.5]).unsqueeze(0).to(device)
    gamma_tx = torch.tensor([0, 0, 0]).unsqueeze(0).to(device)

    ##  Receivers ##
    # locations
    x_rx = torch.tensor([15, 15, 15, 15]).unsqueeze(0).to(device)
    y_rx = torch.tensor([11, 11.5, 12, 12.5]).unsqueeze(0).to(device)
    # properties
    fres_rx = torch.tensor([1, 1, 1, 1]).unsqueeze(0).to(device)
    chi_rx = torch.tensor([0.5, 0.5, 0.5, 0.5]).unsqueeze(0).to(device)
    gamma_rx = torch.tensor([0, 0, 0, 0]).unsqueeze(0).to(device)

    enclosure = {}
    scipy.io.loadmat("..//PhysFad//ComplexEnclosure2.mat", enclosure)
    x_env = torch.tensor(enclosure['x_env']).to(device)
    y_env = torch.tensor(enclosure['y_env']).to(device)
    fres_env = 10 * torch.ones(x_env.shape).to(device)
    chi_env = 50 * torch.ones(x_env.shape).to(device)
    gamma_env = 0 * torch.ones(x_env.shape).to(device)

    RIS_loc = {}
    scipy.io.loadmat("..//PhysFad//ExampleRIS3.mat", RIS_loc)
    x_ris = torch.tensor(RIS_loc['x_ris']).to(device)
    y_ris = torch.tensor(RIS_loc['y_ris']).to(device)
    x_ris_c = x_ris[0].unsqueeze(0).to(device)
    y_ris_c = y_ris[0].unsqueeze(0).to(device)
    parameters = (freq, x_tx, y_tx, fres_tx, chi_tx, gamma_tx,
                  x_rx, y_rx, fres_rx, chi_rx, gamma_rx,
                  x_env, y_env, fres_env, chi_env, gamma_env, x_ris_c, y_ris_c)
    return parameters
def test_configurations_capacity(parameters,ris_configuration,W,device,list_out=False,noise=1):
    H = physfad_model(parameters, ris_configuration,W, device)
    if len(H.shape)==4:
        return capacity_loss(H,torch.ones(H.shape[1],device=device),noise,list_out=list_out),H
    return capacity_loss(H,torch.ones(H.shape[0]),noise,list_out=list_out),H
def capacity_loss(H,P,sigmaN,list_out=False):
    if len(H.shape) == 4:
        H_size = H[0,0, :, :].shape
        number_of_frequencies = H.shape[1]
        rate_freq_list = torch.zeros(len(H),number_of_frequencies)
        if torch.any(~torch.isfinite(H)):
            print("oh no")
        _, S, _ = torch.svd(H, some=False)
        S = S.to(H.device)
        S_N = torch.zeros([S.shape[0],S.shape[1],S.shape[2],S.shape[2]],device =H.device)
        S_N.diagonal(dim1=-2, dim2=-1).copy_(1 + S * S * P.reshape(1, -1, 1) / sigmaN)
        rate_freq_list = torch.log2(torch.det(S_N))
        # for i in range(len(H)): # inside the batch
        #     for f in range(number_of_frequencies):
        #         Sf = torch.squeeze(S[i,f,:])
        #         rate_freq_list[i,f] = torch.log2(torch.prod(1 + torch.norm(Sf)**2 * P[f] / sigmaN))
        if list_out:
            return torch.sum(rate_freq_list,dim=1)
        return torch.sum(rate_freq_list)/H.shape[0]
    else:
        H_size = H[1,:,:].shape
        number_of_frequencies = len(H)
        rate_freq_list = torch.zeros(number_of_frequencies)
        for f in range(number_of_frequencies):
            Hf = torch.squeeze(H[f,:,:])
            _,S,_ = torch.svd(Hf,some=True)
            rate_freq_list[f] = torch.log2(torch.det(torch.diag(1+S*S *P[f] / sigmaN)))
        return torch.sum(rate_freq_list)
def prepeare_env(device):
    freq = torch.tensor(np.linspace(0.9, 1.1, 120));


    ## Configurable Dipole Properties
    ## Transmitters ##
    # locations
    x_tx = torch.tensor([0, 0, 0]).unsqueeze(0).to(device)
    y_tx = torch.tensor([4, 4.5, 5]).unsqueeze(0).to(device)
    # dipole properties
    fres_tx = torch.tensor([1, 1, 1]).unsqueeze(0).to(device)
    chi_tx = torch.tensor([0.5, 0.5, 0.5]).unsqueeze(0).to(device)
    gamma_tx = torch.tensor([0, 0, 0]).unsqueeze(0).to(device)

    ##  Receivers ##
    # locations
    x_rx = torch.tensor([15, 15, 15, 15]).unsqueeze(0).to(device)
    y_rx = torch.tensor([11, 11.5, 12, 12.5]).unsqueeze(0).to(device)
    # properties
    fres_rx = torch.tensor([1, 1, 1, 1]).unsqueeze(0).to(device)
    chi_rx = torch.tensor([0.5, 0.5, 0.5, 0.5]).unsqueeze(0).to(device)
    gamma_rx = torch.tensor([0, 0, 0, 0]).unsqueeze(0).to(device)

    enclosure = {}
    scipy.io.loadmat("..//PhysFad//ComplexEnclosure2.mat", enclosure)
    x_env = torch.tensor(enclosure['x_env']).to(device)
    y_env = torch.tensor(enclosure['y_env']).to(device)
    fres_env = 10 * torch.ones(x_env.shape).to(device)
    chi_env = 50 * torch.ones(x_env.shape).to(device)
    gamma_env = 0 * torch.ones(x_env.shape).to(device)

    RIS_loc = {}
    scipy.io.loadmat("..//PhysFad//ExampleRIS3.mat", RIS_loc)
    x_ris = torch.tensor(RIS_loc['x_ris']).to(device)
    y_ris = torch.tensor(RIS_loc['y_ris']).to(device)
    ris_num_samples = 3
    N_RIS = len(x_ris[0])

    # logical_fres_ris = torch.reshape(torch.tensor(random.choices([True, False], k=ris_num_samples * N_RIS)), [ris_num_samples, N_RIS])
    # resonant_freq =(1.1-0.9)*torch.rand(ris_num_samples, N_RIS)+0.9
    # non_resonant_freq = (5-1.1)*torch.rand(ris_num_samples, N_RIS)+1.1
    # fres_ris = logical_fres_ris * resonant_freq + (~ logical_fres_ris) * non_resonant_freq
    # chi_ris = 0.2*torch.ones([ris_num_samples,x_ris.shape[1]])
    # gamma_ris = 0*torch.ones([ris_num_samples,x_ris.shape[1]])
    x_ris_c = x_ris[0].unsqueeze(0).to(device)
    y_ris_c = y_ris[0].unsqueeze(0).to(device)
    RISConfiguration = np.loadtxt("RandomConfiguration.txt")
    fres_ris_c = torch.tensor(RISConfiguration[0:88]).unsqueeze(0).to(device)
    chi_ris_c = torch.tensor(RISConfiguration[88:176]).unsqueeze(0).to(device)
    gamma_ris_c = torch.tensor(RISConfiguration[176:264]).unsqueeze(0).to(device)

    # RisConfiguration = torch.hstack([fres_ris_c,chi_ris_c,gamma_ris_c])

    # H = GetH(freq,
    #      x_tx,y_tx,fres_tx,chi_tx,gamma_tx,
    #      x_rx,y_rx,fres_rx,chi_rx,gamma_rx,
    #      x_env,y_env,fres_env,chi_env,gamma_env,
    #      x_ris_c,y_ris_c,fres_ris_c,chi_ris_c,gamma_ris_c)




    parameters = (freq, x_tx, y_tx, fres_tx, chi_tx, gamma_tx,
                  x_rx, y_rx, fres_rx, chi_rx, gamma_rx,
                  x_env, y_env, fres_env, chi_env, gamma_env, x_ris_c, y_ris_c)
    # torch.autograd.set_detect_anomaly(True)
    W = get_bessel_w(freq,
                     x_tx, y_tx,
                     x_rx, y_rx,
                     x_env, y_env,
                     x_ris_c, y_ris_c, device)
    return parameters,W

def physfad_channel_optimization(device,starting_inp=None,noise_power = 1,num_of_iterations=150):
    inp_size = 264
    iters = 0
    # num_of_iterations = 150
    if starting_inp is None:
        estOptInp = torch.randn([1, inp_size], requires_grad=True, device=device)
    else:
        estOptInp = starting_inp
    # old_result = torch.from_numpy(np.loadtxt("Physfad_optimal_parameters.txt"))
    # estOptInp = old_result.unsqueeze(0).clone().detach().to(device).requires_grad_(True)
    # scipy.io.savemat("estOptInp.mat", {"estOptInp": estOptInp.cpu().detach().numpy()})

    time_lst = []
    physfad_capacity_lst = []
    Inp_optimizer = torch.optim.Adam([estOptInp], lr=0.1)
    current_loss = torch.Tensor([1])
    parameters,W=prepeare_env(device)
    while (current_loss.item() > -600 and iters < num_of_iterations):
        Inp_optimizer.zero_grad()
        H = physfad_model(parameters, estOptInp ** 2, W, device)
        # scipy.io.savemat("H_python_mat.mat", {"H_python_mat": H.cpu().detach().numpy()})
        # loss = -torch.sum(torch.abs(H[:,0,1]))
        loss = -capacity_loss(H, torch.ones(len(H)), noise_power)
        time_lst.append(datetime.datetime.now())
        physfad_capacity_lst.append(-loss.item())
        loss.backward()
        Inp_optimizer.step()
        # def closure():
        #     optimizer.zero_grad()
        #     pred = net(estOptInp)
        #     loss = -torch.sum(pred)
        #     # grad = torch.autograd.grad(loss,net.parameters(),create_graph=True)
        #     loss.backward()
        #     return loss
        # Inp_optimizer.step(closure)

        if iters % 1 == 0:
            print("iteration #{0} input distance from zero: {1} loss: {2}".format(iters, torch.abs(
                estOptInp).sum().cpu().item(), -loss.item()))
            # if loss.item() < -100:
            #     plt.plot(torch.abs(H[:, 0, 1]).cpu().detach().numpy())
            #     plt.show()

        iters = iters + 1
    return time_lst,physfad_capacity_lst,estOptInp
def random_search_optimization(iteration_limit=300,device='cpu',time_limit=None,noise_power=1, initial_inp = None):
    inp_size = 264
    iters = 0
    # num_of_iterations = 200


    time_lst = []
    random_search_capacity_lst = []
    # Inp_optimizer = torch.optim.Adam([estOptInp], lr=0.01)
    current_loss = 1
    parameters, W = prepeare_env(device)
    from utils import estimate_gradient
    capacity_physfad = lambda x: -capacity_loss(physfad_model(parameters, x ** 2, W, device),
                                                torch.ones(120, device=device), noise_power)
    while ((time_limit is None or time_lst[-1]-time_lst[0]<time_limit) and iters < iteration_limit):
        if iters == 0 and initial_inp != None:
            estOptInp = initial_inp
        else:
            estOptInp = torch.randn([1, inp_size], device=device).cpu().detach().numpy()
        out = capacity_physfad(estOptInp)
        time_lst.append(datetime.datetime.now())
        random_search_capacity_lst.append(-out)
        # def closure():
        #     optimizer.zero_grad()
        #     pred = net(estOptInp)
        #     loss = -torch.sum(pred)
        #     # grad = torch.autograd.grad(loss,net.parameters(),create_graph=True)
        #     loss.backward()
        #     return loss
        # Inp_optimizer.step(closure)

        if iters % 1 == 0:
            print("rand iteration #{0} input distance from zero: {1} loss: {2}".format(iters, np.abs(
                estOptInp).sum(), -out))

        iters = iters + 1
    return time_lst, random_search_capacity_lst
def zeroth_grad_optimization(device,starting_inp=None,noise_power=1,num_of_iterations=200):
    inp_size = 264
    iters = 0
    # num_of_iterations = 200
    if starting_inp is None:
        estOptInp = torch.randn([1, inp_size], requires_grad=True, device=device).cpu().detach().numpy()
    else:
        estOptInp = starting_inp
    epsilon = 0.0001
    m = 4
    lr = 1
    time_lst = []
    physfad_capacity_lst = []
    # Inp_optimizer = torch.optim.Adam([estOptInp], lr=0.01)
    current_loss = 1
    parameters, W = prepeare_env(device)
    from utils import estimate_gradient
    capacity_physfad = lambda x : -capacity_loss(physfad_model(parameters, x ** 2, W, device), torch.ones(120,device=device), noise_power)
    while (current_loss > -300 and iters < num_of_iterations):
        grad_inp = estimate_gradient(capacity_physfad,estOptInp,epsilon,m,device)
        estOptInp = estOptInp - lr * grad_inp
        out = capacity_physfad(estOptInp)
        time_lst.append(datetime.datetime.now())
        physfad_capacity_lst.append(-out)
        # def closure():
        #     optimizer.zero_grad()
        #     pred = net(estOptInp)
        #     loss = -torch.sum(pred)
        #     # grad = torch.autograd.grad(loss,net.parameters(),create_graph=True)
        #     loss.backward()
        #     return loss
        # Inp_optimizer.step(closure)

        if iters % 1 == 0:
            print("iteration #{0} input distance from zero: {1} loss: {2}".format(iters, torch.abs(estOptInp).sum().cpu().item(), -out))

        iters = iters + 1
    return time_lst, physfad_capacity_lst
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print(device)
    zeroth_grad_optimization(device)
    # func = lambda x:(x**2).sum()
    # from utils import estimate_gradient
    # print()
    # x = np.array([1,0])
    # lr = 0.2
    # for i in range(100):
    #     print(x)
    #     grad_x = estimate_gradient(func,x, 0.0000001, 1000)
    #     x = x-lr*grad_x
    # physfad_channel_optimization(device)
    print("Done optimization")

    # print(estOptInp)
    # np.savetxt("Physfad_optimal_parameters.txt", estOptInp[0,:].cpu().detach().numpy())

    # plt.plot(torch.mean(test_pred, dim=0).t().cpu().detach().numpy())
    # plt.legend(["asa,asd"])
    # plt.show()
    # plt.plot(torch.abs(H[:,0,1]))
    # plt.show()
if __name__ == "__main__":
    # cProfile.run('main()')
    main()