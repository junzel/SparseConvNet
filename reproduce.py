import torch
import numpy as numpy
import torch.optim as optim
import random
import torch.nn as nn
import yaml
import pickle
import argparse
import time
import datetime
import os, sys
import shutil, json

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
# from Architectures.CNN import *
# from Architectures.ResNet_3D import *
# from Architectures.SiameseLike_ResNet_oneChannel import *
from Architectures.SparseResNet_3D import *
from DataLoader.DuneNuMuCCDataset_3D import *
# from DataLoader.DuneNuMuCCDataset_oneChannel import *
from Architectures.LossFunctions import *
from lib.listdirFullPath import *
from lib.sortFullPath import *

import pdb

DATA_DIR = '/mnt/sda/dune/data/preprocessed_3D_data'
# DATA_DIR = '/baldig/physicstest/dune/preprocessed_3D_data/'
# DATA_DIR = '/baldig/physicsprojects2/dune/data/cropped_new/'
OUT_DIR = './Results'
SHUFFLE = True
batch_size = 32
epochs = 100

# Read arguments and setup device
parser = argparse.ArgumentParser(description='PyTorch DUNE Direction Regression')
parser.add_argument('--sparse', action='store_true', default=False,
                    help='using submanifold sparse convolutional layers')
parser.add_argument('--name', type=str, default='')
parser.add_argument('--prong', action='store_true', default=False,
                    help='whether train on full-event or prong-only images')
parser.add_argument('--l2', action='store_true', default=False,
                    help='whether add l2 regularization for all params to loss')
parser.add_argument('--uncropped', action='store_true', default=False,
                    help='train on uncropped 100x100x100 data')
parser.add_argument('--type', type=str, default='numucc', choices=['numucc', 'nuecc'])
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using: {}".format(device))

# Creating results/logs saving folder
saving_dir = os.path.join(OUT_DIR, args.name+'_'+str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
if not os.path.isdir(saving_dir):
    os.makedirs(saving_dir)
else:
    print("Folder already exists! Check folder name please!")
    sys.exit()

# Save configurations
shutil.copyfile('train_oneChannel.py', os.path.join(saving_dir, 'train_oneChannel.py'))
# shutil.copyfile('Architectures/ResNet_3D.py', os.path.join(saving_dir, 'ResNet_3D.py'))
shutil.copyfile('Architectures/SparseResNet_3D.py', os.path.join(saving_dir, 'SparseResNet_3D.py'))
with open(os.path.join(saving_dir, 'cofig_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

# Define my_collate for sparse training
if args.sparse:
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
else:
    my_collate = None


# Read file indices
with open('split_paths.pkl', 'rb') as f:
    fileIdx_dict = pickle.load(f)
train_fileIdx_list = fileIdx_dict['train']
val_fileIdx_list = fileIdx_dict['test']  # dict['valid'] is empty

train_dataset = DuneNuMuCCDataset(
                    train_fileIdx_list,
                    os.path.join(DATA_DIR, args.type), 
                    prong=args.prong,
                    transform=None,
                    uncropped=args.uncropped,
                    nu_type=args.type)
val_dataset = DuneNuMuCCDataset(
                    val_fileIdx_list,
                    os.path.join(DATA_DIR, args.type),
                    prong=args.prong,
                    transform=None,
                    uncropped=args.uncropped,
                    nu_type=args.type)

# Initialize data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate)# , num_workers=4)
train_size = train_dataset.__len__()
print("For training process, we have {} events".format(train_size))
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate)# , num_workers=4)
val_size = val_dataset.__len__()
print("For validation process, we have {} events".format(val_size))
train_num_batches = train_dataset.__len__() // batch_size
print("Total training batches: {}".format(train_num_batches))

# Initialize model and optimizers
net = Net().to(device)
meanAngleDiff = meanAngleDiff()
optimizer = optim.Adam(net.parameters(), lr=0.01)# , momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True)

def val_test(model, optimizer, data_loader, how_far, epoch, lowest_loss, date, phase='valid'):
    """Do validation/test at the end of epochs
    Phase='valid': after each several batches,
    Phase='test': at the end of each epoch (no tensorboard recording)
    """
    cos_loss = 0.0
    angle_diff = 0.0
    batch_counter = 0.0
    
    with torch.no_grad():
        for _, batch in enumerate(data_loader):
            inputs = batch['image']
            inputs = inputs.cuda()
            targets = batch['target'].cuda()

            outputs = model(inputs)
            loss = meanAngleDiff.angle_loss_strict(targets, outputs)
            cos_loss += loss.item()
            angle_diff += meanAngleDiff.mean_angle_diff(targets, outputs).item()
            batch_counter += 1
        
        if phase == 'valid':
            writer.add_scalar('Loss/validation loss',
                                cos_loss / batch_counter,
                                how_far)
            writer.add_scalar('AngleDiff/validation angle difference',
                                angle_diff / batch_counter,
                                how_far)
        history['{}_loss'.format(phase)].append((how_far, cos_loss / batch_counter))
        history['{}_angle_diff'.format(phase)].append((how_far, angle_diff / batch_counter))
        
        print('During the %s process, cos_distance: %.3f angle_diff: %.3f' %
                  (phase, cos_loss / batch_counter, angle_diff / batch_counter))

    # Save the best model
    if cos_loss / batch_counter <= lowest_loss:
        lowest_loss = cos_loss / batch_counter
        # torch.save(model, os.path.join(model_dir, '{}_{}.pt'.format(args.name, date)))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': lowest_loss,
            }, os.path.join(model_dir, '{}_{}.ptDict'.format(args.name, date)))
    else: pass

    return lowest_loss


# Set up recorder
log_dir = os.path.join(saving_dir, 'logs'); os.makedirs(log_dir)
model_dir = os.path.join(saving_dir, 'models'); os.makedirs(model_dir)
tensorboard_dir = os.path.join(saving_dir, 'tensorboards'); os.makedirs(tensorboard_dir)
writer = SummaryWriter(os.path.join(tensorboard_dir, '{}_{}'.format(args.name, str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))))
yaml_f = open(os.path.join(log_dir, '{}_training_recorder_{}.yaml'.format(args.name, str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))), 'w')
history = {'train_loss': [], 'train_angle_diff': [], \
            'valid_loss': [], 'valid_angle_diff': [], \
            'test_loss': [], 'test_angle_diff': []
          }
lowest_loss = float("inf")
start_training_date = str(datetime.datetime.now().strftime("%Y%m%d"))

# Start training
for epoch in range(epochs):
    if epoch != 0:
        epoch_time_start_prev = epoch_time_start
    else: 
        epoch_time_start_prev = time.time()
        epoch_time_end = time.time()
    epoch_time_start = time.time()
        
    running_loss1 = 0.0
    running_loss2 = 0.0

    for i_batch, sample_batched in enumerate(train_loader):

        inputs = sample_batched['image']
        if args.sparse:
            inputs = [inputs[0].to(device), inputs[1].to(device)]
        else:
            inputs = inputs.to(device)
        targets = sample_batched['target'].to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = meanAngleDiff.angle_loss_strict(targets, outputs)
        if args.l2:
            reg_lambda = 0.01
            l2_reg = 0
            for W in net.parameters():
                l2_reg += W.norm(2)
            loss += reg_lambda * l2_reg
        loss.backward()
        optimizer.step()

        optimizer.zero_grad()

        running_loss1 += loss.item()
        running_loss2 += meanAngleDiff.mean_angle_diff(targets, outputs).item() # loss.item()

        report_interval = train_num_batches // 50
        if i_batch % report_interval == 0 and i_batch != 0:  # every <report_interval> mini-batches

            writer.add_scalar('Loss/training loss',
                            running_loss1 / report_interval,
                            epoch * len(train_loader) + i_batch)
            writer.add_scalar('AngleDiff/training angle difference',
                            running_loss2 / report_interval,
                            epoch * len(train_loader) + i_batch)
            history['train_loss'].append((epoch * len(train_loader) + i_batch, \
                                            running_loss1 / report_interval))
            history['train_angle_diff'].append((epoch * len(train_loader) + i_batch, \
                                            running_loss2 / report_interval))
            print('[%d, %5d/%5d] cos_distance: %.3f angle_diff: %.3f per_epoch_time: %.2fsec' \
                    % (epoch + 1, i_batch + 1, train_num_batches, \
                    running_loss1 / report_interval, running_loss2 / report_interval, \
                    epoch_time_end - epoch_time_start_prev))
            running_loss1 = 0.0
            running_loss2 = 0.0
            if i_batch % (report_interval * 10) == 0:  # validate after each 1000 mini-batches
                lowest_loss = val_test(net, optimizer, val_loader, \
                                epoch*len(train_loader)+i_batch, epoch, \
                                lowest_loss, start_training_date, phase='valid')
    
    # Test at the end of each epoch
    lowest_loss = val_test(net, optimizer, val_loader, \
                    epoch, epoch, \
                    lowest_loss, start_training_date, phase='test')
    # Save intermediate logs
    yaml.dump(history, yaml_f)

    scheduler.step(loss)
        

# Save training process and trained model
data = yaml.dump(history, yaml_f)
yaml_f.close()
torch.save(net, os.path.join(model_dir, '{}_{}_final.pt'.format(args.name, str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))))
writer.close()

print('Finished Training')

