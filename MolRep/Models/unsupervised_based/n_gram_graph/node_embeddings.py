from __future__ import print_function

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import numpy as np


class CBoW(nn.Module):
    def __init__(self, feature_num, embedding_dim, task_num, task_size_list):
        super(CBoW, self).__init__()
        self.task_num = task_num
        self.embeddings = nn.Linear(feature_num, embedding_dim, bias=False)
        self.layers = nn.ModuleList()
        for task_size in task_size_list:
            self.layers.append(nn.Sequential(
                nn.Linear(embedding_dim, 20),
                nn.ReLU(),
                nn.Linear(20, len(task_size)),
            ))

    def forward(self, x):
        embeds = self.embeddings(x)
        embeds = embeds.sum(1)

        outputs = []
        for layer in self.layers:
            output = layer(embeds)
            outputs.append(output)
        return outputs


def get_data(data_path, padding_size):
    data = np.load(data_path)
    print(data.keys())
    print(data_path)
    adjacent_matrix_list = data['adjacent_matrix_list']
    node_attribute_matrix_list = data['node_attribute_matrix_list']

    molecule_num = adjacent_matrix_list.shape[0]
    print('molecule num\t', molecule_num)

    X_data = []
    Y_label_list = []

    print('adjacent_matrix_list shape: {}\tnode_attribute_matrix_list shape: {}'.format(adjacent_matrix_list.shape, node_attribute_matrix_list.shape))

    for adjacent_matrix, node_attribute_matrix in zip(adjacent_matrix_list, node_attribute_matrix_list):
        assert len(adjacent_matrix) == max_atom_num
        assert len(node_attribute_matrix) == max_atom_num
        for i in range(max_atom_num):
            if sum(adjacent_matrix[i]) == 0:
                break
            x_temp = np.zeros((padding_size, feature_num))
            cnt = 0
            for j in range(max_atom_num):
                if adjacent_matrix[i][j] == 1:
                    x_temp[cnt] = node_attribute_matrix[j]
                    cnt += 1
            x_temp = np.array(x_temp)

            y_temp = []
            atom_feat = node_attribute_matrix[i]
            for s in segmentation_list:
                y_temp.append(atom_feat[s].argmax())

            X_data.append(x_temp)
            Y_label_list.append(y_temp)

    X_data = np.array(X_data)
    Y_label_list = np.array(Y_label_list)
    return X_data, Y_label_list


class GraphDataset(Dataset):
    def __init__(self, mode, K_list, padding_size, segmentation_list):
        self.X_data, self.Y_label_list = [], []
        for i in K_list:
            data_path = '../../datasets/{}/{}_graph.npz'.format(mode, i)
            X_data, Y_label_list = get_data(data_path=data_path, padding_size=padding_size)
            self.X_data.extend(X_data)
            self.Y_label_list.extend(Y_label_list)
        self.X_data = np.array(self.X_data)
        self.Y_label_list = np.array(self.Y_label_list)
        print('data size: ', self.X_data.shape, '\tlabel size: ', self.Y_label_list.shape)
        self.segmentation_list = segmentation_list

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        x_data = self.X_data[idx]
        y_label_list = self.Y_label_list[idx]

        x_data = torch.from_numpy(x_data)
        y_label_list = torch.from_numpy(y_label_list)
        return x_data, y_label_list


def train():
    criterion = nn.CrossEntropyLoss()
    model.train()

    optimal_loss = 1e7
    for epoch in range(epochs):
        train_loss = []

        for batch_id, (x_data, y_actual) in enumerate(train_dataloader):
            x_data = Variable(x_data).float()
            y_actual = Variable(y_actual).long()
            if torch.cuda.is_available():
                x_data = x_data.cuda()
                y_actual = y_actual.cuda()
            optimizer.zero_grad()
            y_predict = model(x_data)

            loss = 0
            for i in range(segmentation_num):
                y_true, y_pred = y_actual[..., i], y_predict[i]
                temp_loss = criterion(y_pred, y_true)
                loss += temp_loss
            loss.backward()
            optimizer.step()
            train_loss.append(loss.data[0])

        train_loss = np.mean(train_loss)
        print('epoch: {}\tloss is: {}'.format(epoch, train_loss))
        if train_loss < optimal_loss:
            optimal_loss = train_loss
            print('Saving model at epoch {}\toptimal loss is {}.'.format(epoch, optimal_loss))
            torch.save(model.state_dict(), weight_file)
    print('For random dimension as {}.'.format(random_dimension))
    return


def test(dataloader):
    model.eval()
    accuracy, total = 0, 0
    for batch_id, (x_data, y_actual) in enumerate(dataloader):
        x_data = Variable(x_data).float()
        y_actual = Variable(y_actual).long()
        if torch.cuda.is_available():
            x_data = x_data.cuda()
            y_actual = y_actual.cuda()
        y_predict = model(x_data)

        for i in range(segmentation_num):
            y_true, y_pred = y_actual[..., i].cpu().data.numpy(), y_predict[i].cpu().data.numpy()
            y_pred = y_pred.argmax(1)
            accuracy += np.sum(y_true == y_pred)
            total += y_pred.shape[0]
    accuracy = 1. * accuracy / total
    print('Accuracy: {}'.format(accuracy))

    print('For random dimension as {}.'.format(random_dimension))

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='delaney')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--running_index', type=int, default=0)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    mode = args.mode
    epochs = args.epochs
    running_index = args.running_index
    seed = args.seed

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        cudnn.benchmark = True

    random_dimension_list = [50, 100]

    if mode in ['qm8', 'qm9']:
        feature_num = 32
        segmentation_list = [range(0, 10), range(10, 17), range(17, 24), range(24, 30), range(30, 32)]
    else:
        feature_num = 42
        segmentation_list = [range(0, 10), range(10, 17), range(17, 24), range(24, 30), range(30, 36),
                             range(36, 38), range(38, 40), range(40, 42)]

    if mode in ['hiv']:
        max_atom_num = 100
        padding_size = 10
    elif 'pcba' in mode or 'clintox' in mode:
        max_atom_num = 100
        padding_size = 6
    else:
        max_atom_num = 55
        padding_size = 6

    segmentation_list = np.array(segmentation_list)
    segmentation_num = len(segmentation_list)

    test_list = [running_index]
    train_list = filter(lambda x: x not in test_list, np.arange(5))
    print('training list: {}\ttest list: {}'.format(train_list, test_list))

    for random_dimension in random_dimension_list:
        weight_file = 'model_weight/{}/{}/{}_CBoW_non_segment.pt'.format(mode, running_index, random_dimension)

        model = CBoW(feature_num=feature_num, embedding_dim=random_dimension,
                     task_num=segmentation_num, task_size_list=segmentation_list)
        if torch.cuda.is_available():
            model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)

        train_dataset = GraphDataset(mode, K_list=train_list, segmentation_list=segmentation_list, padding_size=padding_size)
        test_dataset = GraphDataset(mode, K_list=test_list, segmentation_list=segmentation_list, padding_size=padding_size)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True)

        train()

        test(train_dataloader)
        test(test_dataloader)
        print()
        print()
        print()
