import torch as T
from sklearn.decomposition import PCA
import numpy as np
from ChannelMatrixEvaluation import capacity_loss
import scipy.io
## DEBUG!!
from ChannelMatrixEvaluation import test_configurations_capacity, get_configuration_parameters
import torch
## DEBUG!!
class RISDataset(T.utils.data.Dataset):

    def __init__(self, RIS_config_file,H_realizations_file,H_capacity_file, output_size =120,output_shape =(4,3),calculate_capacity = False,only_fres=False, m_rows=None, device=T.device("cpu")):
        # ris_configs_np = np.loadtxt(RIS_config_file,delimiter=",", dtype=np.float32)
        # H_realiz_np = np.loadtxt(H_realizations_file,delimiter=",", dtype=np.float32)
        enclosure = {}
        scipy.io.loadmat(RIS_config_file, enclosure)
        ris_configs_np = enclosure["RISConfiguration"]
        enclosure = {}
        scipy.io.loadmat(H_realizations_file,enclosure)
        H_realiz_np = enclosure["sampled_Hs"].reshape(-1,output_size*output_shape[0]*output_shape[1])

        ## DEBUG!!
        # physfad_capacity, physfad_H = test_configurations_capacity(get_configuration_parameters(device),
        #                                                            torch.Tensor(ris_configs_np[0, :]).unsqueeze(0).to(device),
        #                                                            device)
        ## DEBUG!!
        # self.pca = PCA(n_components=20)
        # self.pca.fit(H_realiz_np)
        # self.reduced_dimensionality_realizations = self.pca.fit_transform(H_realiz_np)
        if only_fres:
            self.x_data = T.tensor(ris_configs_np[:,0:88], \
                               dtype=T.float32).to(device)
        else:
            self.x_data = T.tensor(ris_configs_np, \
                                   dtype=T.float32).to(device)
        self.y_data = T.tensor(H_realiz_np, \
                               dtype=T.float32).to(device)
        if calculate_capacity:
            print("calculating capacity")
            self.y_capacity = capacity_loss(T.tensor(H_realiz_np,dtype=T.float32,device=device).reshape((-1,output_size,output_shape[0],output_shape[1])),torch.ones(output_size,device=device),1,list_out=True)
            np.savetxt(H_capacity_file,self.y_capacity.cpu().detach().numpy(),delimiter=",")
            print("Done calculating")
        else:
            self.y_capacity = T.tensor(np.loadtxt(H_capacity_file,delimiter=",",dtype=np.float32),dtype=T.float32,device=device)
        self.dataset_changed = False
        # self.y_data = T.tensor(self.reduced_dimensionality_realizations, \
        #                        dtype=T.float32).to(device)

    def add_new_items(self,X,Y,Y_capacity):
        if X is None:
            return
        if torch.any(~torch.isfinite(X)):
            print("non finite value detected")
        self.x_data = T.vstack([self.x_data,X])
        self.y_data = T.vstack([self.y_data,Y])
        self.y_capacity = T.hstack([self.y_capacity,Y_capacity])
        self.dataset_changed = True
    def save_dataset(self,RIS_config_file,H_realiz_file,capacity_file):
        if capacity_file is not None:
            np.savetxt(capacity_file, self.y_capacity, delimiter=",")
        if RIS_config_file is not None:
            scipy.io.savemat(RIS_config_file, {"RISConfiguration": self.x_data.cpu().detach().numpy()})
        if H_realiz_file is not None:
            scipy.io.savemat(H_realiz_file, {"sampled_Hs": self.y_data.cpu().detach().numpy()})
            # ## DEBUG
            # enclosure = {}
            # scipy.io.loadmat(RIS_config_file, enclosure)
            # ris_config = enclosure["RISConfiguration"]
            # print(ris_config)
            # if not torch.equal(self.x_data,T.tensor(ris_config, \
            #                                         dtype=T.float32).to("cpu")):
            #     print("data not equal after save")
            # train_ds_debug = T.tensor(H_realiz_np, dtype=T.float32).to("cpu")
            # if torch.any(~torch.isfinite(train_ds_debug)):
            #     print("non finite value detected")
            # train_ldr_debug = T.utils.data.DataLoader(train_ds_debug, batch_size=32, shuffle=True)
            # if torch.any(~torch.isfinite(train_ldr_debug)):
            #     print("non finite value detected")
            #
            # ## End DEBUG

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        preds = self.x_data[idx, :]  # or just [idx]
        gt_capacity = self.y_capacity[idx]
        gt = self.y_data[idx, :]

        return (preds, gt_capacity, gt)  # tuple of two matrices
