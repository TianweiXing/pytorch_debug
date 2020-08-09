import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="1";

from torchsummary import summary

import sys
import pickle
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CLEVR, collate_data, transform
from model import MACNetwork

import embedding as ebd


import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.embed = nn.Embedding(400001, 300)
        # Embedding layer: loading weights
        embedding_matrix = ebd.load()
        print(embedding_matrix.shape)
        self.embed.weight.data = torch.Tensor(embedding_matrix)
        self.embed.weight.requires_grad = False
        
        self.lstm1 = nn.LSTM(300, 128, 1, batch_first=True)
        self.lstm2 = nn.LSTM(128, 128, 1, batch_first=True)
        
        
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 28)


    def forward(self, question):
        batch_size = question.size()[0]
        
        embed = self.embed(question)
#         print(embed.shape)
        lstm_out, _ = self.lstm1(embed)
        lstm_out = F.dropout(lstm_out, 0.5)
#         print(lstm_out.shape)
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = F.dropout(lstm_out, 0.5)
#         print(lstm_out.shape)
        
        x = lstm_out[:,-1]
#         print(x.shape)
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


from torch.utils.data import DataLoader
import torch
from torch.utils import data

class My_Data2(data.Dataset):
    def __init__(self,  split='train', transform=None):
        
        processed_test_data_path = 'test_data.npz'
    
        npzfile = np.load(processed_test_data_path)
#         print(npzfile.files)
#         self.data_s_split = npzfile['s_' + split]#$[3011:3031]
        self.data_a_split = npzfile['a_' + split]#[3011:3031]
        self.data_q_split = npzfile['q_' + split]#[3011:3031]
        
        # adjust dimension
        self.data_a_split = self.data_a_split.argmax(1)
#         self.data_s_split = np.expand_dims(self.data_s_split, -1)  
#         self.data_s_split = np.swapaxes(self.data_s_split,1,2)
#         self.data_s_split = np.expand_dims(self.data_s_split, -1)
        self.split = split  # train or val

    def __getitem__(self, index):
#         data_s = self.data_s_split[index]
        data_q = self.data_q_split[index]
        data_a = self.data_a_split[index]
        return data_q, len(data_q), data_a
#         return data_s, data_q, len(data_q), data_a
    
    def __len__(self):
        return len(self.data_a_split)
    
    
def train(epoch):
#     clevr = CLEVR(sys.argv[1], transform=transform)
    training_set = My_Data2(split='val')
    train_set = DataLoader(
        training_set, batch_size=batch_size, num_workers=1
#         , collate_fn=collate_data
    )

    dataset = iter(train_set)
    pbar = tqdm(dataset)
    moving_loss = 0
#     acc_accumulate = 0

    net.train(True)
    for iter_id, (question, q_len, answer) in enumerate(pbar):
        
#         image = image.type(torch.FloatTensor) # change data type: double to float
        q_len = q_len.tolist()
        question = question.type(torch.LongTensor)
        
        question, answer = (
            question.to(device),
            answer.to(device),
        )

        net.zero_grad()
        output = net(question)
        loss = criterion(output, answer)
        loss.backward()
        optimizer.step()
        correct = output.detach().argmax(1) == answer
        correct = torch.tensor(correct, dtype=torch.float32).sum() / batch_size
        
        # correct is the acc for current batch, moving_loss is the acc for previous batches
        if moving_loss == 0:
            moving_loss = correct
        else:
            moving_loss = (moving_loss * iter_id + correct)/(iter_id+1)
#             moving_loss = moving_loss * 0.99 + correct * 0.01

        pbar.set_description(
            'Epoch: {}; Loss: {:.5f}; Current_Acc: {:.5f}; Total_Acc: {:.5f}'.format(
                epoch + 1, loss.item(), correct, moving_loss
            )
        )



def valid(epoch):
#     clevr = CLEVR(sys.argv[1], 'val', transform=None)
    training_set = My_Data2(split='val')
    valid_set = DataLoader(
        training_set, batch_size=batch_size, num_workers=1
#         , collate_fn=collate_data
    )
    
    dataset = iter(valid_set)

    net.train(False)
    family_correct = Counter()
    family_total = Counter()
    loss_total = 0
    
    with torch.no_grad():
        for  question, q_len, answer in tqdm(dataset):
            
            family = [1]*len(question)
#             image = image.type(torch.FloatTensor) # change data type: double to float
            q_len = q_len.tolist()
            question = question.type(torch.LongTensor)
            
            question = question.to(device)

            output = net(question)
            loss = criterion(output, answer.to(device))
            
            loss_total = loss_total + loss
            correct = output.detach().argmax(1) == answer.to(device)
            for c, fam in zip(correct, family):
                if c:
                    family_correct[fam] += 1
                family_total[fam] += 1
                

    print(
        'Avg Acc: {:.5f}; Avg Loss: {:.5f}'.format(
            sum(family_correct.values()) / sum(family_total.values()),
            loss_total / sum(family_total.values())
        )
    )

    print('%d / %d'%(sum(family_correct.values()), sum(family_total.values())))
    return sum(family_correct.values()) / sum(family_total.values())






if __name__ == '__main__':

    embedding_matrix = ebd.load()
    print(embedding_matrix.shape)
    # embedding_matrix = ebd.load()
    print('Size of word embedding matrix: ',embedding_matrix.shape)
    num_words = 400001
    embedding_dim = 300
    seq_length = 31#data_q_valid.shape[1] 

    num_hidden_lstm = 128
    output_dim =128
    dropout_rate = 0.5

    sen_dim = 77
    sen_win_len = 1800 
    sen_channel = 1
    num_feat_map = 64

    num_classes = 28#data_a_valid.shape[1]
    
    
    
    lstm_net = Net()
    
    
    batch_size = 64
    n_epoch = 20
    dim = 512

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    net = Net().to(device)
    # net_running = Net().to(device)
    # accumulate(net_running, net, 0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    
    
    acc_best = 0.0

    for epoch in range(100):
    # for epoch in range(n_epoch):
        print('==========%d epoch =============='%(epoch))
        train(epoch)
        acc = valid(epoch) # inference on: validation dataset