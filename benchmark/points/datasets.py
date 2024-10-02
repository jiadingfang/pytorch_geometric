import os.path as osp

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet


def get_dataset(num_points):
    name = 'ModelNet10'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    pre_transform = T.NormalizeScale()
    transform = T.SamplePoints(num_points)

    train_dataset = ModelNet(path, name='10', train=True, transform=transform,
                             pre_transform=pre_transform)
    test_dataset = ModelNet(path, name='10', train=False, transform=transform,
                            pre_transform=pre_transform)

    return train_dataset, test_dataset


if __name__ == '__main__':
    train_dataset, test_dataset = get_dataset(num_points=1024)

    sample = train_dataset[0]
    print(sample) # Data(pos=[1024, 3], y=[1])

    print(train_dataset)
    print(test_dataset)