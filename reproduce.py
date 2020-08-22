import torch
import numpy as np
import torch.optim as optim
import random
import torch.nn as nn
import argparse
import time
import h5py
import datetime
import os, sys, pdb

from torch.utils.data import Dataset, DataLoader
import sparseconvnet as scn

batch_size = 32
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using: {}".format(device))

class SparseResNet(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.sparseModel = scn.Sequential(
            scn.SubmanifoldConvolution(3, 1, 64, 7, False),  # sscn(dimension, nIn, nOut, filter_size, bias)
            scn.BatchNormReLU(64),
            scn.MaxPooling(3, 3, 2),  # MaxPooling(dimension, pool_size, pool_stride)
            scn.SparseResNet(3, 64, [  # SpraseResNet(dimension, nInputPlanes, layers=[])
                        ['b', 64, 2, 1],  # [block_type, nOut, rep, stride]
                        ['b', 64, 2, 1],
                        ['b', 128, 2, 2],
                        ['b', 128, 2, 2],
                        ['b', 256, 2, 2],
                        ['b', 256, 2, 2],
                        ['b', 512, 2, 2],
                        ['b', 512, 2, 2]]),
            scn.SparseToDense(3, 256))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.spatial_size= self.sparseModel.input_spatial_size(torch.LongTensor([1, 1]))
        self.spatial_size = torch.LongTensor([101, 101])
        self.inputLayer = scn.InputLayer(3,self.spatial_size, mode=3)

    def forward(self, x):
        pdb.set_trace()
        x = self.inputLayer(x)
        x = self.sparseModel(x)
        x = self.avgpool(x)
        # x = x.view(-1, 64)
        x = torch.flatten(x, 1)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.ResNet = SparseResNet()

        self.fc = nn.Linear(512 * 1, 4096)
        self.relu = nn.ReLU(inplace=True)
        self.linear = nn.Linear(4096, 3)

    def forward(self, x):
        top = self.ResNet(x)
        top = self.fc(top)
        top = self.relu(top)
        out = self.linear(top)

        return out


class DuneNuMuCCDataset(Dataset):
    """Dune NuMuCC dataset. """

    def __init__(self, transform=None, nu_type='nuecc'):
        """
        Args:
            file_list (list of string): list of names of the data files.
            root_dir (string): Directory with all the data (images+targets).
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_path = '/mnt/sda/dune/preprocessed_3D_data/{}/3D_images/0.h5'.format(nu_type)
        self.target_path = '/mnt/sda/dune/preprocessed_3D_data/{}/prong_direction/0.h5'.format(nu_type)
        self.reco_path = '/mnt/sda/dune/preprocessed_3D_data/{}/prong_reco_direction/0.h5'.format(nu_type)
        self.data = {}
        # with h5py.File(self.image_path, 'r') as f_image, \
        #         h5py.File(self.target_path, 'r') as f_target, \
        #         h5py.File(self.reco_path, 'r') as f_reco:
        #     images = f_image['data'][...]
        #     targets = f_target['data'][...]
        #     recos = f_reco['data'][...]
        images = np.zeros([483, 1, 100, 100, 100])
        targets = np.ones([483, 3])
        recos = np.ones([483, 3])
        self.data['image'] = images
        self.data['target'] = targets
        self.data['reco'] = recos
        # DATA_DIR = '/baldig/physicstest/dune/preprocessed_3D_data/'
    
    def __len__(self):
        return len(self.data['target'])

    def __getitem__(self, index):
        # check nan
        if np.isnan(self.data['target'][index]).sum() > 0:
            # pdb.set_trace()
            # img = self.data['image'][index]
            pass
        
        return_data = {
            'image': self.data['image'][index], \
            'target': self.data['target'][index], \
            'reco': self.data['reco'][index] \
                        }

        return return_data


# Define my_collate for sparse training
def my_collate(batch):
    """Process the batch to fit into submanifold sparse format"""
    msgs = [item['image'] for item in batch]
    target = [item['target'] for item in batch]

    locations = []  # Only one channel/view
    features = []
    for batchIdx, msg in enumerate(msgs):
        for y in range(msg.shape[1]):
            for x in range(msg.shape[2]):
                for z in range(msg.shape[3]):
                    if msg[0, x, y, z] != 0.0:
                        locations.append([z, y, x, batchIdx])
                        features.append([1])
    locations = torch.LongTensor(locations)
    features = torch.FloatTensor(features)


    target = torch.FloatTensor(target)

    return_data = {
        'image': [locations, features], \
        'target': target}
    return return_data

# Initialize dataset
train_dataset = DuneNuMuCCDataset()
# Initialize data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate)# , num_workers=4)
train_size = train_dataset.__len__()
print("For training process, we have {} events".format(train_size))


# Initialize model and optimizers
net = Net().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)# , momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True)

# Start training
for epoch in range(epochs):
    if epoch != 0:
        epoch_time_start_prev = epoch_time_start
    else: 
        epoch_time_start_prev = time.time()
        epoch_time_end = time.time()
    epoch_time_start = time.time()
        
    running_loss = 0.0
    counter = 0.0
    for i_batch, sample_batched in enumerate(train_loader):

        inputs = sample_batched['image']
        inputs = [inputs[0].to(device), inputs[1].to(device)]
        targets = sample_batched['target'].to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(targets, outputs)

        loss.backward()
        optimizer.step()

        optimizer.zero_grad()

        running_loss += loss.item()
        counter += 1.0

    print("Loss: {}".format(running_loss / counter))

    scheduler.step(loss)

print('Finished Training')

