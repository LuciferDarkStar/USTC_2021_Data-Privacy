import torch
import copy
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models.Nets import CNNMnist


class Server():
    def __init__(self, args, w):
        self.args = args
        self.clients_update_w = []
        self.clients_loss = []
        self.model = CNNMnist(args=args).to(args.device)
        self.model.load_state_dict(w)

    def FedAvg(self):
        if self.args.mode == 'plain':
            update_w_avg = copy.deepcopy(self.clients_update_w[0])
            for k in update_w_avg.keys():
                for i in range(1, len(self.clients_update_w)):
                    update_w_avg[k] += self.clients_update_w[i][k]
                update_w_avg[k] = torch.div(update_w_avg[k], len(self.clients_update_w))
                self.model.state_dict()[k] += update_w_avg[k]
        elif self.args.mode == 'DP':
            # 设置参数
            c = 0.001
            ParameterSigma = 0.1

            update_w_avg = copy.deepcopy(self.clients_update_w[0])
            for k in update_w_avg.keys():
                for i in range(1, len(self.clients_update_w)):
                    MaxW = max(1, torch.norm(self.clients_update_w[i][k], 2) / c)  # 找出最大的值
                    update_w_avg[k] += self.clients_update_w[i][k] / MaxW  # 求和
                update_w_avg[k] += torch.normal(0, ParameterSigma * c,
                                                update_w_avg[k].shape, device=self.args.device)  # 根据高斯机制添加噪音
                update_w_avg[k] = torch.div(update_w_avg[k], len(self.clients_update_w))  # 聚合模型本地更新
                self.model.state_dict()[k] += update_w_avg[k]
        elif self.args.mode == 'Paillier':
            update_w_avg = copy.deepcopy(self.clients_update_w[0])
            for k in update_w_avg.keys():
                for i in range(1, len(self.clients_update_w)):
                    for n in range(len(update_w_avg[k])):
                        update_w_avg[k][n] += self.clients_update_w[i][k][n]  # 求和
                for n in range(len(update_w_avg[k])):
                    update_w_avg[k][n] /= len(self.clients_update_w)  # 同态加密聚合梯度
            return copy.deepcopy(update_w_avg), sum(self.clients_loss) / len(self.clients_loss)

        return copy.deepcopy(self.model.state_dict()), sum(self.clients_loss) / len(self.clients_loss)

    def test(self, datatest):
        self.model.eval()

        # testing
        test_loss = 0
        correct = 0
        data_loader = DataLoader(datatest, batch_size=self.args.bs)
        for idx, (data, target) in enumerate(data_loader):
            if self.args.gpu != -1:
                data, target = data.cuda(), target.cuda()
            log_probs = self.model(data)

            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()

            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)
        return accuracy, test_loss
