import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from models.Nets import CNNMnist
from phe import paillier
import copy

KeyPub, KeyPriv = paillier.generate_paillier_keypair()


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class Client():

    def __init__(self, args, dataset=None, idxs=None, w=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.model = CNNMnist(args=args).to(args.device)
        self.model.load_state_dict(w)

    def train(self):
        w_old = copy.deepcopy(self.model.state_dict())
        net = copy.deepcopy(self.model)

        net.train()

        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

        w_new = net.state_dict()

        update_w = {}
        if self.args.mode == 'plain':
            for k in w_new.keys():
                update_w[k] = w_new[k] - w_old[k]
        elif self.args.mode == 'DP':
            for k in w_new.keys():
                update_w[k] = w_new[k] - w_old[k]
        elif self.args.mode == 'Paillier':
            for k in w_new.keys():
                update_w[k] = (w_new[k] - w_old[k]).flatten(start_dim=0).tolist()  # 用w_new-w_old代替
                for j in range(len(update_w[k])):
                    update_w[k][j] = KeyPub.encrypt(update_w[k][j])  # 用公共密钥进行加密

        return update_w, sum(batch_loss) / len(batch_loss)

    def update(self, w_glob):
        if self.args.mode == 'plain':
            self.model.load_state_dict(w_glob)
        elif self.args.mode == 'DP':
            self.model.load_state_dict(w_glob)
        elif self.args.mode == 'Paillier':
            for k in w_glob.keys():
                for i in range(len(w_glob[k])):
                    w_glob[k][i] = KeyPriv.decrypt(w_glob[k][i])  # 用私钥进行解密
                Final_tensor = torch.Tensor(w_glob[k]).to(self.args.device)
                Final_tensor = Final_tensor.reshape(self.model.state_dict()[k].shape)  # 类型转换回来
                self.model.state_dict()[k] += Final_tensor

