import csv
import serial
import os 
import argparse
import torch 
import torch.nn.functional as F
from torch import nn 
from progressbar import *
from torch.optim import Adam
import util
import random
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from tqdm import tqdm
from datetime import date

class TPALSTM(nn.Module):

    def __init__(self, input_size, output_horizon, hidden_size, obs_len, n_layers):
        super(TPALSTM, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, \
                    bias=True, batch_first=True) # output (batch_size, obs_len, hidden_size)
        self.hidden_size = hidden_size
        self.filter_num = 32
        self.filter_size = 1
        self.output_horizon = output_horizon
        self.attention = TemporalPatternAttention(self.filter_size, \
            self.filter_num, obs_len-1, hidden_size)
        self.linear = nn.Linear(hidden_size, output_horizon)
        self.n_layers = n_layers

    def forward(self, x):
        batch_size, obs_len = x.size()
        x = x.view(batch_size, obs_len, 1)
        xconcat = self.relu(self.hidden(x))
        # x = xconcat[:, :obs_len, :]
        # xf = xconcat[:, obs_len:, :]
        H = torch.zeros(batch_size, obs_len-1, self.hidden_size)
        ht = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        ct = ht.clone()
        for t in range(obs_len):
            xt = xconcat[:, t, :].view(batch_size, 1, -1)
            out, (ht, ct) = self.lstm(xt, (ht, ct))
            htt = ht.permute(1, 0, 2)
            htt = htt[:, -1, :]
            if t != obs_len - 1:
                H[:, t, :] = htt
        H = self.relu(H)
        
        # reshape hidden states H
        H = H.view(-1, 1, obs_len-1, self.hidden_size)
        new_ht = self.attention(H, htt)
        ypred = self.linear(new_ht)
        return ypred

class TemporalPatternAttention(nn.Module):

    def __init__(self, filter_size, filter_num, attn_len, attn_size):
        super(TemporalPatternAttention, self).__init__()
        self.filter_size = filter_size
        self.filter_num = filter_num
        self.feat_size = attn_size - self.filter_size + 1
        self.conv = nn.Conv2d(1, filter_num, (attn_len, filter_size))
        self.linear1 = nn.Linear(attn_size, filter_num)
        self.linear2 = nn.Linear(attn_size + self.filter_num, attn_size)
        self.relu = nn.ReLU()
    
    def forward(self, H, ht):
        _, channels, _, attn_size = H.size()
        new_ht = ht.view(-1, 1, attn_size)
        w = self.linear1(new_ht) # batch_size, 1, filter_num 
        conv_vecs = self.conv(H)
        
        conv_vecs = conv_vecs.view(-1, self.feat_size, self.filter_num)
        conv_vecs = self.relu(conv_vecs)

        # score function
        w = w.expand(-1, self.feat_size, self.filter_num)
        s = torch.mul(conv_vecs, w).sum(dim=2)
        alpha = torch.sigmoid(s)
        new_alpha = alpha.view(-1, self.feat_size, 1).expand(-1, self.feat_size, self.filter_num)
        v = torch.mul(new_alpha, conv_vecs).sum(dim=1).view(-1, self.filter_num)
        
        concat = torch.cat([ht, v], dim=1)
        new_ht = self.linear2(concat)
        return new_ht

parser = argparse.ArgumentParser()
parser.add_argument("--num_epoches", "-e", type=int, default=1000)
parser.add_argument("--step_per_epoch", "-spe", type=int, default=2)
parser.add_argument("-lr", type=float, default=1e-3)
parser.add_argument("--n_layers", "-nl", type=int, default=3)
parser.add_argument("--hidden_size", "-hs", type=int, default=24)
parser.add_argument("--seq_len", "-sl", type=int, default=7)
parser.add_argument("--num_obs_to_train", "-not", type=int, default=1)
parser.add_argument("--num_results_to_sample", "-nrs", type=int, default=10)
parser.add_argument("--show_plot", "-sp", action="store_true")
parser.add_argument("--run_test", "-rt", action="store_true")
parser.add_argument("--standard_scaler", "-ss", action="store_true")
parser.add_argument("--log_scaler", "-ls", action="store_true")
parser.add_argument("--mean_scaler", "-ms", action="store_true")
parser.add_argument("--max_scaler", "-max", action="store_true")
parser.add_argument("--batch_size", "-b", type=int, default=64)
args = parser.parse_args(args=['-e','100','-spe','24','-nl','1','-not','168','-sl','144','-sp','-rt','-max'])
model = torch.load(".//model//save_model_spe_24.pt")
model.eval()


def pred():

  data=pd.read_csv(".//get_data.csv",names=[0,1])
  seq_len = args.seq_len
  obs_len = args.num_obs_to_train
  num_ts=1
  yte = data[0][-obs_len:].to_numpy().reshape((1, -1))
  y_test = yte.reshape((num_ts, -1))
  yscaler = None
  if args.standard_scaler:
      yscaler = util.StandardScaler()
  elif args.log_scaler:
      yscaler = util.LogScaler()
  elif args.mean_scaler:
      yscaler = util.MeanScaler()
  elif args.max_scaler:
      yscaler = util.MaxScaler()
  if yscaler is not None:
      y_test = yscaler.fit_transform(y_test)
  y_test = torch.from_numpy(y_test).float()
  ypred = model(y_test)
  ypred = ypred.data.numpy()
  if yscaler is not None:
      ypred = yscaler.inverse_transform(ypred)
  ypred = ypred.ravel()
  dec = ypred[0]-ypred[-1]
  dec = int(ypred[0]/dec*len(ypred)*15)
  #print(str(ypred[-1])+","+str(dec)+"\n")
  return  ypred[-1],dec

SerialIn = serial.Serial("COM6",9600)

try:
   while True:
       while SerialIn.in_waiting:
           data_in = SerialIn.readline()
           #print(data_in)
           data_raw = data_in.decode('utf-8').split()[0]
           #print(data_raw)
           data_raw=data_raw.split(',')
           #print(data_raw)
           mois=float(data_raw[0])
           temper =float(data_raw[1])
           with open('get_data.csv', mode='a' , newline='') as afile:
             data_writer = csv.writer(afile, delimiter=',')
             data_writer.writerow([mois,temper])
           print(mois,temper)
           y,d = pred()
           string = (str(y)+","+str(d)+"\n")
           SerialIn.write(string.encode('utf-8'))
           #print(pred())

except KeyboardInterrupt:
   SerialIn.close()