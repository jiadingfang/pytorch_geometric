import argparse

import torch
import torch.nn.functional as F
from points.datasets import get_dataset
from points.train_eval import run
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq
from torch.utils.data import Subset, random_split

from torch_geometric.nn import PointNetConv, fps, global_max_pool, radius_graph
from torch_geometric.profile import rename_profile_file

from points.scannet_dataset import ScanNetDataset


class Net(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        nn = Seq(Lin(3, 64), ReLU(), Lin(64, 64))
        self.conv1 = PointNetConv(local_nn=nn)

        nn = Seq(Lin(67, 128), ReLU(), Lin(128, 128))
        self.conv2 = PointNetConv(local_nn=nn)

        nn = Seq(Lin(131, 256), ReLU(), Lin(256, 256))
        self.conv3 = PointNetConv(local_nn=nn)

        self.lin1 = Lin(256, 256)
        self.lin2 = Lin(256, 256)
        self.lin3 = Lin(256, num_classes)

    def forward(self, pos, batch):
        radius = 0.2
        edge_index = radius_graph(pos, r=radius, batch=batch)
        x = F.relu(self.conv1(None, pos, edge_index))

        idx = fps(pos, batch, ratio=0.5)
        x, pos, batch = x[idx], pos[idx], batch[idx]

        radius = 0.4
        edge_index = radius_graph(pos, r=radius, batch=batch)
        x = F.relu(self.conv2(x, pos, edge_index))

        idx = fps(pos, batch, ratio=0.25)
        x, pos, batch = x[idx], pos[idx], batch[idx]

        radius = 1
        edge_index = radius_graph(pos, r=radius, batch=batch)
        x = F.relu(self.conv3(x, pos, edge_index))

        x = global_max_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--lr_decay_step_size', type=int, default=50)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()

    # # model net dataset
    # train_dataset, test_dataset = get_dataset(num_points=1024)
    # model = Net(train_dataset.num_classes)
    # run(train_dataset, test_dataset, model, args.epochs, args.batch_size, args.lr,
    #     args.lr_decay_factor, args.lr_decay_step_size, args.weight_decay,
    #     args.inference, args.profile, args.bf16, args.compile)

    # if args.profile:
    #     rename_profile_file('points', PointNetConv.__name__)

    # scannet dataset
    train_dataset = ScanNetDataset("/share/data/pals/fjd/pointcloud_label_data", split='train', num_points=2048, xyz_only=True)
    test_dataset = ScanNetDataset("/share/data/pals/fjd/pointcloud_label_data", split='test', num_points=2048, xyz_only=True)

    # split train dataset into train and val into 90% and 10%
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    print('train dataset length:', len(train_dataset))
    print('val dataset length:', len(val_dataset))
    print('test dataset length:', len(test_dataset))


    model = Net(train_dataset.dataset.num_classes)

    # # settings: padding => removing => more points (2048)

    # # train on train dataset and test on test dataset => 64 => 70
    # print('train on train dataset and test on test dataset')
    # run(train_dataset, test_dataset, model, args.epochs, args.batch_size, args.lr,
    #     args.lr_decay_factor, args.lr_decay_step_size, args.weight_decay,
    #     args.inference, args.profile, args.bf16, args.compile)

    # train on train dataset and validate on val dataset => 67 => 75
    print('train on train dataset and validate on val dataset')
    run(train_dataset, val_dataset, model, args.epochs, args.batch_size, args.lr,
        args.lr_decay_factor, args.lr_decay_step_size, args.weight_decay,
        args.inference, args.profile, args.bf16, args.compile)

    # # overfitting => 86 => 99
    # print('overfitting')
    # run(val_dataset, val_dataset, model, args.epochs, args.batch_size, args.lr,
    #     args.lr_decay_factor, args.lr_decay_step_size, args.weight_decay,
    #     args.inference, args.profile, args.bf16, args.compile)