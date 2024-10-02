from torch_geometric.data import Data
from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import glob
from tqdm import tqdm
import json


# def read_poointcloud_data(scan_id, root):
#     points_path = f"{root}/{scan_id}_pointcloud.npy"
#     labels_path = f"{root}/{scan_id}_labels.txt"
#     points = torch.from_numpy(np.load(points_path)).float()
#     labels = torch.from_numpy(np.loadtxt(labels_path)).long()
#     return points, labels

# source: https://kaldir.vc.in.tum.de/scannet_benchmark/labelids_all.txt
# source: https://kaldir.vc.in.tum.de/scannet_benchmark/labelids.txt

# setup random seed
NUMPY_RANDOM_SEED = 42

class ScanNetDataset(Dataset):

    def __init__(self, root, split='train', xyz_only=True, num_points=1024, fixed_random_seed=True):
        self.root = root
        self.split = split
        self.xyz_only = xyz_only
        self.num_points = num_points
        self.fixed_random_seed = fixed_random_seed

        # get scannet label mapping
        # label_mapping_path = '/share/data/old-pals/fjd/pytorch_geometric/benchmark/points/labelids.txt'
        # label_mapping_path = '/share/data/old-pals/fjd/pytorch_geometric/benchmark/points/labelids_all.txt'
        label_mapping_path = '/share/data/old-pals/fjd/pytorch_geometric/benchmark/points/nyu_40_classes_enumerated.txt'
        self.scannet_label_mapping = {}
        with open(label_mapping_path, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                line = line.strip()
                if len(line) == 0:
                    continue
                # label_id, label_name = line.split('\t')
                self.scannet_label_mapping[line] = idx

        instance_label_to_sementic_label_path = '/share/data/old-pals/fjd/referit3d/referit3d/data/mappings/scannet_instance_class_to_semantic_class.json'
        self.instance_label_to_sementic_label = {}
        with open(instance_label_to_sementic_label_path, 'r') as f:
            instance_label_to_sementic_label = json.load(f)
            for instance_label, semantic_label in instance_label_to_sementic_label.items():
                self.instance_label_to_sementic_label[instance_label] = semantic_label

        # get sr3d train/test split
        sr3d_test_path = '/share/data/old-pals/fjd/Transcrib3D/data/referit3d/sr3d_test.csv'
        sr3d_train_path = '/share/data/old-pals/fjd/Transcrib3D/data/referit3d/sr3d_train.csv'
        sr3d_test = pd.read_csv(sr3d_test_path)
        sr3d_train = pd.read_csv(sr3d_train_path)

        test_list = sorted(list(set(sr3d_test['scan_id'].values.tolist())))
        train_list = sorted(list(set(sr3d_train['scan_id'].values.tolist())))
        # print('train_list length:', len(train_list))
        # print('test_list length:', len(test_list))

        if split == 'train':
            self.scan_ids = train_list
        else:
            self.scan_ids = test_list

        # get pointcloud and label files
        self.pointcloud_files = []
        self.label_files = []
        for scan_id in tqdm(self.scan_ids):
            scan_path = Path(self.root) / scan_id
            pointcloud_files = sorted(glob.glob(str(scan_path / '*_pointcloud.npy')))
            label_files = sorted(glob.glob(str(scan_path / '*_label.txt')))
            # assert len(pointcloud_files) == len(label_files), f"pointcloud_files: {len(pointcloud_files)}, label_files: {len(label_files)}"

            # if number of points is less than self.num_points, remove the scan
            valid_pointcloud_files = []
            for pointcloud_file in pointcloud_files:
                points = np.load(pointcloud_file)
                if points.shape[0] >= self.num_points:
                    valid_pointcloud_files.append(pointcloud_file)
            pointcloud_files = valid_pointcloud_files
            label_files = [pointcloud_file.replace('_pointcloud.npy', '_label.txt') for pointcloud_file in pointcloud_files]

            assert len(pointcloud_files) == len(label_files), f"pointcloud_files: {len(pointcloud_files)}, label_files: {len(label_files)}"

            self.pointcloud_files.extend(pointcloud_files)
            self.label_files.extend(label_files)

            
    def __len__(self):
        return len(self.pointcloud_files)
    
    @property
    def num_classes(self):
        return len(self.scannet_label_mapping)
    
    def __getitem__(self, idx):
        pointcloud_path = self.pointcloud_files[idx]
        label_path = self.label_files[idx]
        points = torch.from_numpy(np.load(pointcloud_path)).float()

        assert points.size(0) >= self.num_points, f"points.size(0): {points.size(0)}, num_points: {self.num_points}"
        if self.fixed_random_seed:
            np.random.seed(NUMPY_RANDOM_SEED)
        idx = np.random.choice(points.size(0), self.num_points, replace=False)
        points = points[idx]
        
        # if points.size(0) >= self.num_points:
        #     # randomly downsample the pointcloud
        #     if self.fixed_random_seed:
        #         np.random.seed(NUMPY_RANDOM_SEED)
        #     idx = np.random.choice(points.size(0), self.num_points, replace=False)
        #     points = points[idx]
        # else:
        #     # pad the pointcloud
        #     num_pad = self.num_points - points.size(0)
        #     idx = np.random.choice(points.size(0), num_pad, replace=True)
        #     points = torch.cat([points, points[idx]], dim=0)

        if self.xyz_only:
            points = points[:, :3]

        # read one-line label file
        with open(label_path, 'r') as f:
            label_name = f.readline().strip()
            label_name = self.instance_label_to_sementic_label[label_name]
        assert label_name in self.scannet_label_mapping 
        label = self.scannet_label_mapping[label_name]
        label = torch.tensor([label]).long()
        # return points, label
        return Data(pos=points, y=label)
        # return Data(pos=points), semantic_label


if __name__=="__main__":
    scannet_root = "/share/data/old-pals/fjd/pointcloud_label_data" # low-res
    # scannet_root = "/share/data/old-pals/fjd/Transcrib3D/data/pointcloud_label_data_hi-res" # hi-res
    scannet_dataset = ScanNetDataset(scannet_root, split='train', num_points=1024, xyz_only=True)
    print('train dataset length:', len(scannet_dataset))
    scannet_dataset_test = ScanNetDataset(scannet_root, split='test', num_points=1024, xyz_only=True)
    print('test dataset length:', len(scannet_dataset_test))

    sample = scannet_dataset[0]
    print(sample)

    # check the datapoint sizes in the dataset
    data_sizes = []
    label_names = []
    for i in tqdm(range(len(scannet_dataset))):
        data, label = scannet_dataset[i]
        # data_sizes.append(data.pos.size(0))
        # label_names.append(label)

    # breakpoint()
    
    # len(set(instance_label_names)) # 531
    # len(set(semantic_label_names)) # 40

    # print(min(data_sizes)) # low-res: 21; high-res: 415
    # print(max(data_sizes)) # low-res: 190241; high-res: 5267848
    # print(np.mean(data_sizes)) # low-res: 3989.4; high-res: 99817.2


    



