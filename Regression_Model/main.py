# house_price.py
# predict price from AC, sq ft, style, nearest school
# PyTorch 1.7.0-CPU Anaconda3-2020.02  Python 3.7.6
# Windows 10

import numpy as np
import torch.optim
from matplotlib import pyplot as plt
import time
import torch as T

device = T.device("cpu")  # apply to Tensor or Module


# -----------------------------------------------------------

class RISDataset(T.utils.data.Dataset):

    def __init__(self, RIS_config_file,H_realizations_file, m_rows=None):
        ris_configs_np = np.loadtxt(RIS_config_file,delimiter=",", dtype=np.float32)
        H_realiz_np = np.loadtxt(H_realizations_file,delimiter=",", dtype=np.float32)


        self.x_data = T.tensor(ris_configs_np, \
                               dtype=T.float32).to(device)
        self.y_data = T.tensor(H_realiz_np, \
                               dtype=T.float32).to(device)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        preds = self.x_data[idx, :]  # or just [idx]
        price = self.y_data[idx, :]
        return (preds, price)  # tuple of two matrices


# -----------------------------------------------------------

class Net(T.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hid1 = T.nn.Linear(135, 1000)  # 8-(10-10)-1
        self.dropout1 = T.nn.Dropout(0.1)
        self.hid2 = T.nn.Linear(1000, 500)
        self.dropout2 = T.nn.Dropout(0.1)
        self.oupt = T.nn.Linear(500, 101)

        T.nn.init.xavier_uniform_(self.hid1.weight)
        T.nn.init.zeros_(self.hid1.bias)
        T.nn.init.xavier_uniform_(self.hid2.weight)
        T.nn.init.zeros_(self.hid2.bias)
        T.nn.init.xavier_uniform_(self.oupt.weight)
        T.nn.init.zeros_(self.oupt.bias)

    def forward(self, x):
        z = T.relu(self.hid1(x))
        z = self.dropout1(z)
        z = T.relu(self.hid2(z))
        z = self.dropout2(z)
        z = self.oupt(z)  # no activation

        return z


# -----------------------------------------------------------

# def accuracy(model, ds, pct):
#     # assumes model.eval()
#     # percent correct within pct of true house price
#     n_correct = 0;
#     n_wrong = 0
#
#     for i in range(len(ds)):
#         (X, Y) = ds[i]  # (predictors, target)
#         with T.no_grad():
#             oupt = model(X)  # computed price
#
#         abs_delta = np.abs(oupt.item() - Y.item())
#         max_allow = np.abs(pct * Y.item())
#         if abs_delta < max_allow:
#             n_correct += 1
#         else:
#             n_wrong += 1
#
#     acc = (n_correct * 1.0) / (n_correct + n_wrong)
#     return acc


# -----------------------------------------------------------

# def accuracy_quick(model, dataset, pct):
#     # assumes model.eval()
#     n = len(dataset)
#     X = dataset[0:n][0]  # all predictor values
#     Y = dataset[0:n][1]  # all target prices
#     with T.no_grad():
#         oupt = model(X)  # all computed prices
#
#     max_deltas = T.abs(pct * Y)  # max allowable deltas
#     abs_deltas = T.abs(oupt - Y)  # actual differences
#
#     results = abs_deltas < max_deltas  # [[True, False, . .]]
#     acc = T.sum(results, dim=0).item() / n  # dim not needed
#     return acc


# -----------------------------------------------------------

# def baseline_acc(ds, pct):
#     # linear regression model accuracy using just sq. feet
#     # y = 1.9559x + 0.0987 (from separate program)
#     n_correct = 0;
#     n_wrong = 0
#     for i in range(len(ds)):
#         (X, Y) = ds[i]  # (predictors, target)
#         x = X[1].item()  # sq feet predictor
#         y = 1.9559 * x + 0.0987  # computed
#
#         abs_delta = np.abs(oupt.item() - Y.item())
#         max_allow = np.abs(pct * Y.item())
#         if abs_delta < max_allow:
#             n_correct += 1
#         else:
#             n_wrong += 1
#
#     acc = (n_correct * 1.0) / (n_correct + n_wrong)
#     return acc
def MSE_accuracy(estimations,ground_truth):
    return np.mean((estimations.detach().numpy()-ground_truth.detach().numpy())**2)

# -----------------------------------------------------------

def main():
    # 0. get started
    T.manual_seed(4)  # representative results
    np.random.seed(4)

    # 1. create DataLoader objects
    print("Collecting RIS configuration ")
    train_RIS_file = "..\\Data\\RISConfiguration.txt"
    train_H_file = "..\\Data\\H_realizations.txt"
    train_ds = RISDataset(train_RIS_file,train_H_file)  # all 200 rows

    test_RIS_file = "..\\Data\\Test_RISConfiguration.txt"
    test_H_file = "..\\Data\\Test_H_realizations.txt"
    test_ds = RISDataset(test_RIS_file, test_H_file)  # all 200 rows
    # test_file = ".\\houses_test.txt"
    # test_ds = RISDataset(test_file)  # all 40 rows

    batch_size = 5
    train_ldr = T.utils.data.DataLoader(train_ds,
                                        batch_size=batch_size, shuffle=True)
    test_ldr = T.utils.data.DataLoader(test_ds,
                                        batch_size=batch_size, shuffle=True)
    # 2. create network
    net = Net().to(device)

    # 3. train model
    max_epochs = 200
    ep_log_interval = 20
    lrn_rate = 0.0005
    training_mode = True
    loss_func = T.nn.MSELoss()
    # optimizer = T.optim.SGD(net.parameters(), lr=lrn_rate)
    optimizer = T.optim.Adam(net.parameters(), lr=lrn_rate)

    print("\nbat_size = %3d " % batch_size)
    print("loss = " + str(loss_func))
    print("optimizer = Adam")
    print("max_epochs = %3d " % max_epochs)
    print("lrn_rate = %0.4f " % lrn_rate)

    print("\nStarting training with saved checkpoints")
    if training_mode:
        net.train()  # set mode
        for epoch in range(0, max_epochs):
            T.manual_seed(1 + epoch)  # recovery reproducibility
            epoch_loss = 0  # for one full epoch

            for (batch_idx, batch) in enumerate(train_ldr):
                (X, Y) = batch  # (predictors, targets)
                optimizer.zero_grad()  # prepare gradients
                oupt = net(X)  # predicted prices
                # print(torch.sum(oupt).item())

                loss_val = loss_func(oupt, Y)  # avg per item in batch
                epoch_loss += loss_val.item()  # accumulate avgs
                loss_val.backward()  # compute gradients
                optimizer.step()  # update wts


            if epoch % ep_log_interval == 0:
                print("epoch = %4d   loss = %0.4f" % \
                      (epoch, epoch_loss))
                plt.plot(oupt.t().detach().numpy(),'r')
                plt.plot(Y.t().detach().numpy(),'b--')
                plt.legend(["output","ground truth"])
                plt.show()
                count = 0
                test_MSE = 0
                for (test_batch_idx, test_batch) in enumerate(test_ldr):
                    (X_test,Y_test) = test_batch
                    test_output = net(X_test)
                    test_MSE += MSE_accuracy(test_output,Y_test)
                    count = count+1
                test_MSE = test_MSE/(count*batch_size)
                print("MSE for test is "+str(test_MSE))
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
    if training_mode:
        torch.save(net.state_dict(),".\\Models\\Main_model.pt")
    else:
        net.load_state_dict(torch.load(".\\Models\\Main_model.pt"))

    print("Done ")
    net.train()
    # run gradiant descent to find best input configuration

    (X, _) = next(iter(train_ldr))  # (predictors, targets)
    estOptInp = torch.randn(X.size())
    Inp_optimizer = torch.optim.Adam([estOptInp],lr=0.0000001)
    current_loss = torch.Tensor([1])
    iters = 0
    num_of_iterations = 500
    while (current_loss.item() > -300 and iters < num_of_iterations):
        def closure():
            optimizer.zero_grad()
            pred = net(estOptInp)
            loss = -torch.sum(pred)
            # grad = torch.autograd.grad(loss,net.parameters(),create_graph=True)
            loss.backward()
            return loss
        Inp_optimizer.step(closure)
        iters = iters+1
        if iters % 30 == 0:
            current_loss = closure()
            test_pred = net(estOptInp)
            print(torch.sum(test_pred).item())
            plt.plot(test_pred.t().detach().numpy())
            plt.legend(["asa,asd"])

            plt.show()
    print("done optimization")
    # 4. evaluate model accuracy
    # print("\nComputing model accuracy")
    # net.eval()
    # acc_train = accuracy(net, train_ds, 0.10)
    # print("Accuracy (within 0.10) on train data = %0.4f" % \
    #       acc_train)

    # acc_test = accuracy(net, test_ds, 0.10)
    # print("Accuracy (within 0.10) on test data  = %0.4f" % \
    #       acc_test)

    # base_acc_train = baseline_acc(train_ds, 0.10)
    # print("%0.4f" % base_acc_train)  # 0.7000
    # base_acc_test = baseline_acc(test_ds, 0.10)
    # print("%0.4f" % base_acc_test)   # 0.7000

    # 5. make a prediction
    # print("\nPredicting price for AC=no, sqft=2300, ")
    # print(" style=colonial, school=kennedy: ")
    # unk = np.array([[-1, 0.2300, 0, 0, 1, 0, 1, 0]],
    #                dtype=np.float32)
    # unk = T.tensor(unk, dtype=T.float32).to(device)
    #
    # with T.no_grad():
    #     pred_price = net(unk)
    # pred_price = pred_price.item()  # scalar
    # str_price = \
    #     "${:,.2f}".format(pred_price * 1000000)
    # print(str_price)

    # 6. save final model (state_dict approach)
    # print("\nSaving trained model state")
    # fn = ".\\Models\\houses_model.pth"
    # T.save(net.state_dict(), fn)

    # saved_model = Net()
    # saved_model.load_state_dict(T.load(fn))
    # use saved_model to make prediction(s)

    print("\nEnd House price demo")


if __name__ == "__main__":
    main()

