import torch
import scipy.special as scp
import numpy as np
import scipy.io
import math
import datetime

def besselj(order,z):
    return 1
def bessely(order,z):
    return torch.special.bessel_y0()
def besselh(order,kind=2,z=0,scale=0):
    # return besselj(0,z)-torch.tensor([1j])*bessely(0,z);
    return scp.hankel2(order, z)
def get_bessel_w(freq,
         x_tx,y_tx,
         x_rx,y_rx,
         x_env,y_env,
         x_ris,y_ris,device):
    k = 2 * torch.pi * freq
    x = torch.cat([x_tx, x_rx, x_env, x_ris], 1)
    y = torch.cat([y_tx, y_rx, y_env, y_ris], 1)

    N_T = len(x_tx[0])
    N_R = len(x_rx[0])
    N_E = len(x_env[0])
    N_RIS = len(x_ris[0])
    N = N_T + N_R + N_E + N_RIS
    H = torch.zeros([len(freq), N_R, N_T], dtype=torch.complex64, device=device)
    pi = torch.pi
    W = torch.zeros([len(freq),N,N],dtype=torch.complex64,device=device)
    for f in range(len(freq)):
        x_diff = torch.zeros([N, N], dtype=torch.float64, device=device)
        y_diff = torch.zeros([N, N], dtype=torch.float64, device=device)
        for l in range(N):
            xl_vec = x[0, l] * torch.ones([1, N], dtype=torch.float64, device=device)
            yl_vec = y[0, l] * torch.ones([1, N], dtype=torch.float64, device=device)
            x_diff[l, :] = x - xl_vec
            y_diff[l, :] = y - yl_vec
        BesselInp = k[f] * torch.sqrt(x_diff ** 2 + y_diff ** 2)
        # BesselInp = torch.sqrt(x_diff**2+y_diff**2)
        BesselOut = torch.tensor(besselh(0, 2, BesselInp.cpu().numpy()), device=device)
        W[f] = 1j * (k[f] ** 2 / 4) * BesselOut
    return W
def GetH_batched(freq, W_full,
         x_tx,y_tx,fres_tx,chi_tx,gamma_tx,
         x_rx,y_rx,fres_rx,chi_rx,gamma_rx,
         x_env,y_env,fres_env,chi_env,gamma_env,
         x_ris,y_ris,fres_ris,chi_ris,gamma_ris,device):
    # print("batched Physfad")
    epsilon = 0.00000001
    k = 2 * torch.pi * freq
    # x = torch.cat([x_tx, x_rx, x_env, x_ris],1)
    # y = torch.cat([y_tx, y_rx, y_env, y_ris],1)
    batch_size = fres_ris.shape[0]
    fres = torch.cat([fres_tx.repeat(batch_size,1), fres_rx.repeat(batch_size,1), fres_env.repeat(batch_size,1), fres_ris], 1)
    chi = torch.cat([chi_tx.repeat(batch_size,1), chi_rx.repeat(batch_size,1), chi_env.repeat(batch_size,1), chi_ris], 1)
    gamma = torch.cat([gamma_tx.repeat(batch_size,1), gamma_rx.repeat(batch_size,1), gamma_env.repeat(batch_size,1), gamma_ris], 1)

    N_T = len(x_tx[0])
    N_R = len(x_rx[0])
    N_E = len(x_env[0])
    N_RIS = len(x_ris[0])
    N = N_T + N_R + N_E + N_RIS
    pi = torch.pi
    k2 = (torch.pow(k, 2)).repeat(batch_size,1).to(device)
    two_pi = 2 * pi
    two_pi_freq = (two_pi * freq).repeat(batch_size,1).to(device)
    two_pi_freq2 = torch.pow(two_pi_freq, 2)

    chi2 = torch.pow(chi, 2)+epsilon
    two_pi_fres2 = torch.pow((two_pi * fres), 2)

    inv_alpha = (two_pi_fres2.unsqueeze(2) - two_pi_freq2.unsqueeze(1)) / (chi2.unsqueeze(2)) + 1j * ((
                k2.unsqueeze(1) / 4) + two_pi_freq.unsqueeze(1) * gamma.unsqueeze(2) / chi2.unsqueeze(2))
    inv_alpha = inv_alpha.type(torch.complex64)
    if torch.any(~torch.isfinite(inv_alpha)):
        print("oh no1")
    W = W_full.clone().repeat(batch_size,1,1,1)
    # width = W.size(0)
    Mask = torch.eye(W.size(2)).repeat(batch_size,len(freq), 1, 1).bool()
    W[Mask] = inv_alpha.permute(0,2,1).reshape(-1)
    W_diag_elem = torch.diagonal(W, dim1=-2, dim2=-1)
    W_diag_matrix = torch.zeros(W.shape, dtype=torch.complex64, device=device)
    W_diag_matrix.diagonal(dim1=-2, dim2=-1).copy_(W_diag_elem)
    if torch.any(~torch.isfinite(W_diag_matrix)):
        print("oh no2")
    V = torch.linalg.solve(W, W_diag_matrix)
    if torch.any(~torch.isfinite(V)):
        print("oh no3")
    H = V[:,:, N_T: (N_T + N_R), 0: N_T]
    if torch.any(~torch.isfinite(H)):
        print("oh no4")
    return H

def GetH(freq, W_full,
         x_tx,y_tx,fres_tx,chi_tx,gamma_tx,
         x_rx,y_rx,fres_rx,chi_rx,gamma_rx,
         x_env,y_env,fres_env,chi_env,gamma_env,
         x_ris,y_ris,fres_ris,chi_ris,gamma_ris,device):

    if fres_ris.shape[0]!=1:
        return GetH_batched(freq, W_full,
             x_tx,y_tx,fres_tx,chi_tx,gamma_tx,
             x_rx,y_rx,fres_rx,chi_rx,gamma_rx,
             x_env,y_env,fres_env,chi_env,gamma_env,
             x_ris,y_ris,fres_ris,chi_ris,gamma_ris,device)
    # print("normal Physfad")
    epsilon = 0.00000001
    k=2*torch.pi*freq
    # x = torch.cat([x_tx, x_rx, x_env, x_ris],1)
    # y = torch.cat([y_tx, y_rx, y_env, y_ris],1)
    fres = torch.cat([fres_tx, fres_rx, fres_env, fres_ris],1)
    chi = torch.cat([chi_tx, chi_rx, chi_env, chi_ris],1)
    gamma = torch.cat([gamma_tx, gamma_rx, gamma_env, gamma_ris],1)

    N_T   = len(x_tx[0])
    N_R   = len(x_rx[0])
    N_E   = len(x_env[0])
    N_RIS = len(x_ris[0])
    N = N_T + N_R + N_E + N_RIS
    H = torch.zeros([len(freq),N_R,N_T],dtype=torch.complex64,device=device)
    pi = torch.pi
    k2 = (torch.pow(k, 2)).to(device)
    two_pi = 2 * pi
    two_pi_freq = (two_pi * freq).to(device)
    two_pi_freq2 = torch.pow(two_pi_freq, 2)
    chi2 = torch.pow(chi[0, :], 2)+epsilon
    gamma_ = gamma[0,:]
    two_pi_fres2 = torch.pow((two_pi * fres[0, :]), 2)

    inv_alpha = (two_pi_fres2.unsqueeze(1) - two_pi_freq2.unsqueeze(0)) / (chi2.unsqueeze(1)) + 1j * ((k2.unsqueeze(0) / 4) + two_pi_freq.unsqueeze(0) * gamma_.unsqueeze(1) / chi2.unsqueeze(1))
    inv_alpha = inv_alpha.type(torch.complex64)
    W = W_full.clone()
    # width = W.size(0)
    Mask = torch.eye(W.size(1)).repeat(len(freq), 1, 1).bool()
    W[Mask] = inv_alpha.T.reshape(-1)
    W_diag_elem = torch.diagonal(W,dim1=-2,dim2=-1)
    W_diag_matrix = torch.zeros(W.shape,dtype=torch.complex64,device=device)
    W_diag_matrix.diagonal(dim1=-2,dim2=-1).copy_(W_diag_elem)
    scipy.io.savemat("W_mat.mat", {"W_mat": W.cpu().detach().numpy()})

    V = torch.linalg.solve(W, W_diag_matrix)
    H = V[:,N_T: (N_T + N_R), 0: N_T]

    # for f in range(len(freq)):
        # x_diff = torch.zeros([N,N],dtype=torch.float64,device=device)
        # y_diff = torch.zeros([N, N],dtype=torch.float64,device=device)
        # for l in range(N):
        #     xl_vec = x[0,l] * torch.ones([1, N],dtype=torch.float64,device=device)
        #     yl_vec = y[0,l] * torch.ones([1, N],dtype=torch.float64,device=device)
        #     x_diff[l,:] = x - xl_vec
        #     y_diff[l,:] = y - yl_vec
        # BesselInp = k[f]*torch.sqrt(x_diff**2+y_diff**2)
        # BesselOut = torch.tensor(besselh(0,2,BesselInp.cpu().numpy()),device=device)
        # W = 1j * (k[f] ** 2 / 4) * BesselOut
        # W = W_full[f].clone()

        # start_time = datetime.datetime.now()
        # inv_alpha = (two_pi_fres2 - two_pi_freq2[f]) / (chi2) + 1j * (k2[f] / 4) + two_pi_freq[f] * gamma_ / chi2
        # width = W.size(0)
        # W.as_strided([width], [width + 1]).copy_(inv_alpha[:,f])
        # W.as_strided([width], [width + 1]).copy_(inv_alpha)
        # for i in range(0,N):
            # diagonal entries of W are the inverse polarizabilities
            # a =
            # b =

            # inv_alpha = a
            # W[i, i] = inv_alpha[i]
            # if i==(N-1):
            #     scipy.io.savemat("inv_alpha.mat", {"inv_alpha": inv_alpha.cpu().detach().numpy()})



        #%Invert W and extract H
        # np.savetxt("W_mat.txt", W.cpu().detach().numpy())
        # scipy.io.savemat("W_mat.mat", {"W_mat": W.cpu().detach().numpy()})
        # Winv = torch.inverse(W)
        # V = torch.matmul(torch.diag(torch.diag(W)),Winv)
        # after_mat_calc = datetime.datetime.now()
        # calc_time = after_mat_calc - start_time
        # print("matrix calculation time {0}".format(str(calc_time)))
        # V = torch.linalg.solve(W, torch.diag(torch.diag(W)))
        # V = torch.linalg.solve(W[f], torch.diag(torch.diag(W[f])))
        # scipy.io.savemat("W_mat.mat", {"W_mat": W})
        # inv_time = datetime.datetime.now() - after_mat_calc
        # print("matrix inversion time {0}".format(str(inv_time)))
        # H[f,:,:] = V[N_T: (N_T + N_R), 0: N_T]

    return H



#             ((2*pi*fres(ii))^2-(2*pi*freq(ff))^2)/
#             ((2 * torch.pi * fres[0, i]) ** 2 - (2 * torch.pi * freq[f]) ** 2)
#
#             (chi(ii)^2) + 1i*(((k(ff)^2)/4) + 2*pi*freq(ff)*gamma(ii)/(chi(ii)^2));
#             (chi[0,i] ** 2) + 1j * (((k[f] ** 2) / 4) + 2 * torch.pi * freq[f] * gamma[0,i] / (chi[0,i] ** 2))
#
# inv_alpha = ((2 * torch.pi * fres[0,i]) ** 2 - (2 * torch.pi * freq[f]) ** 2)/
#              (chi[0,i] ** 2) + 1j * ((k[f] ** 2 / 4) + 2 * torch.pi * freq[f] * gamma[0,i] / (chi[0,i] ** 2))