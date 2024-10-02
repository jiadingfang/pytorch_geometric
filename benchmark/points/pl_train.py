import argparse
import datetime

import torch
import torch.nn.functional as F

from torch_geometric.loader import DataLoader

from points.point_net import Net as PointNet
from points.point_cnn import Net as PointCNN
from points.scannet_dataset import ScanNetDataset

import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch import loggers




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--model', type=str, default='pointnet', choices=['pointnet', 'pointcnn'])
    parser.add_argument('--num-points', type=int, default=1024)
    args = parser.parse_args()
    return args


class calculate_accuracy(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        correct = 0
        total = 0
        # breakpoint()
        device = pl_module.device
        with torch.no_grad():
            for data in trainer.val_dataloaders:
                data = data.to(device)
                outputs = pl_module(data)
                _, predicted = torch.max(outputs, 1)
                total += data.y.size(0)
                correct += (predicted == data.y).sum().item()
        accuracy = correct / total
        print(f'Accuracy: {accuracy:.4f}')
        trainer.logger.log_metrics({'val_acc': accuracy})



class PLPointNet(L.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.model = PointNet(num_classes)

    def forward(self, data):
        return self.model(data.pos, data.batch)
    
    def training_step(self, batch, batch_idx):
        # breakpoint()
        out = self(batch)
        loss = F.nll_loss(out, batch.y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # breakpoint()
        out = self(batch)
        loss = F.nll_loss(out, batch.y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    

class PLPointCNN(L.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.model = PointCNN(num_classes)

    def forward(self, data):
        return self.model(data.pos, data.batch)
    
    def training_step(self, batch, batch_idx):
        # breakpoint()
        out = self(batch)
        loss = F.nll_loss(out, batch.y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # breakpoint()
        out = self(batch)
        loss = F.nll_loss(out, batch.y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    

if __name__=="__main__":

    args = parse_args()
    print(args)

    scannet_root = "/share/data/old-pals/fjd/pointcloud_label_data" # low-res
    # scannet_root = "/share/data/old-pals/fjd/Transcrib3D/data/pointcloud_label_data_hi-res" # hi-res
    train_dataset = ScanNetDataset(scannet_root, split='train', num_points=args.num_points)
    test_dataset = ScanNetDataset(scannet_root, split='test', num_points=args.num_points)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=2)

    # model
    if args.model == 'pointnet':
        model = PLPointNet(train_dataset.num_classes)
    elif args.model == 'pointcnn':
        model = PLPointCNN(train_dataset.num_classes)
    else:
        raise ValueError(f"Unknown model: {args.model}, choose from ['pointnet', 'pointcnn']")

    # get timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # logger
    if args.model == 'pointnet':
        logger = loggers.TensorBoardLogger('points/logs', name='pointnet', version=timestamp)
    elif args.model == 'pointcnn':
        logger = loggers.TensorBoardLogger('points/logs', name='pointcnn', version=timestamp)
    else:
        raise ValueError(f"Unknown model: {args.model}, choose from ['pointnet', 'pointcnn']")

    # trainer
    trainer = L.Trainer(
        max_epochs=100,
        devices=1,
        callbacks=[calculate_accuracy()],
        logger=logger,
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    trainer.validate(model=model, val_dataloaders=test_loader)


    # setting: pointnet, removing, points (4096) => 70
    # setting: pointcnn, removing, points (2048) => 73
    # setting: pointcnn, removing, points (4096) => 72
    # setting: pointcnn, removing, points (1024) => 68