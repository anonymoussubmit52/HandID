import sys
import random
import torch
import torch.nn as nn
import torch.utils.data as Data
import pandas as pd
import numpy as np
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
from torchsampler import ImbalancedDatasetSampler
from h_transformer_1d import HTransformer1D


epoch_input=sys.argv[1]
ts = datetime.now()
EPOCH = int(epoch_input)
BATCH_SIZE = 16
LR = 1e-5
DOWNLOAD_MNIST = False
svdnumber = 100
max_accuracy = 0
feat_dim = 150
Tclass=2
Lclass=9 
seed = 33

random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


# Provide the path of the preprocessed csv dataset
dataset_train = pd.read_csv('')
dataset_train = dataset_train.sample(frac=1)
dataset_test = pd.read_csv('')
dataset_test = dataset_test.sample(frac=1)

XH = dataset_train.iloc[:, :feat_dim].to_numpy()
XH = np.reshape(XH, (-1,1,feat_dim))
yH = dataset_train.iloc[:, -2].to_numpy()
zH = dataset_train.iloc[:, -1].to_numpy()
yH = yH - 1
zH = zH - 1
XL = dataset_test.iloc[:, :feat_dim].to_numpy()
XL = np.reshape(XL, (-1,1,feat_dim))
yL = dataset_test.iloc[:, -2].to_numpy()
yL = yL - 1

yH[yH != 1] = 0
yL[yL != 1] = 0

trainX, testX, trainy, testy = train_test_split(XH, yH,test_size=0.2, random_state=42)
trainX1, testX1, trainy1, testy1 = train_test_split(XL, yL, test_size=0.2, random_state=42)
trainX2, testX2, trainz, testz = train_test_split(XH, zH,test_size=0.2, random_state=42)

train_x = torch.from_numpy(trainX)
train_x = torch.nn.functional.normalize(train_x)
train_y = torch.from_numpy(trainy)
train_x = train_x.float()
train_y = train_y.float()
train_z = torch.from_numpy(trainz)
train_z = train_z.float()
train_dataset = TensorDataset(train_x,train_y,train_z)
validate_x = torch.from_numpy(testX)
validate_x = torch.nn.functional.normalize(validate_x)
validate_y = torch.from_numpy(testy)
validate_x = validate_x.float()
validate_y = validate_y.float()
validate_data = TensorDataset(validate_x,validate_y)
test_x = torch.from_numpy(testX1)
test_x = torch.nn.functional.normalize(test_x)
test_y = torch.from_numpy(testy1)
test_x = test_x.float()
test_y = test_y.float()
test_data = TensorDataset(test_x,test_y)


mark_test_y = test_y.size(dim=0)
mark_validate_y = train_x.size(dim=0)*0.2
print(train_z.size())

# Data Loader for easy mini-batch return in training
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, 
    sampler=ImbalancedDatasetSampler(train_dataset),
    batch_size=BATCH_SIZE, 
)
validate_loader = torch.utils.data.DataLoader(dataset=validate_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

class AdversarialNet(nn.Module):
    def __init__(self):
        super(AdversarialNet, self).__init__()
        self.featureex = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        
        self.Conv1d_4 = nn.Conv1d(1, 1, kernel_size=4, stride=1, padding=1)
        self.Conv1d_8 = nn.Conv1d(1, 1, kernel_size=8, stride=1, padding=1)
        self.Conv1d_16 =nn.Conv1d(1, 1, kernel_size=16, stride=1, padding=1)

        self.classify = nn.Sequential(
            nn.Linear(30400, Tclass)
        )

        self.disclam = nn.Sequential(
            nn.Linear(30400, Lclass)
        )

  
    def forward(self, x):
        h_1 = torch.cat((self.Conv1d_4(x), self.Conv1d_8(x), self.Conv1d_16(x)), 2)
        h_2 = torch.cat((self.Conv1d_4(h_1), self.Conv1d_8(h_1), self.Conv1d_16(h_1)), 2)
        h_3 = torch.cat((self.Conv1d_4(h_2), self.Conv1d_8(h_2), self.Conv1d_16(h_2)), 2)
        f = self.featureex(h_3)
        features = f.view(f.size(0), -1)
        o_c = self.classify(features)
        o_d = self.disclam(features)
        return o_c, o_d

adversarial_model = AdversarialNet()

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!" )
  adversarial_model = nn.DataParallel(adversarial_model)

adversarial_model.cuda()

print('begin...',file=open(str(ts)+"loggan.txt", "a"))

if torch.cuda.device_count() > 1:
    optimizer_classifier = torch.optim.Adam(list(adversarial_model.module.featureex.parameters())+list(adversarial_model.module.classify.parameters()), lr=LR)
    optimizer_domain = torch.optim.Adam(list(adversarial_model.module.featureex.parameters())+list(adversarial_model.module.disclam.parameters()), lr=LR)
else:
    optimizer_classifier = torch.optim.Adam(list(adversarial_model.featureex.parameters())+list(adversarial_model.classify.parameters()), lr=LR)
    optimizer_domain = torch.optim.Adam(list(adversarial_model.featureex.parameters())+list(adversarial_model.disclam.parameters()), lr=LR)
loss_func = nn.CrossEntropyLoss()

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

for epoch in range(EPOCH):
    train_step = 0
    for step, sample in enumerate(train_loader):
        if torch.cuda.is_available():
            x = sample[0].cuda()
            y = sample[1].cuda()
            z = sample[2].cuda()
            y_pred, z_pred = adversarial_model(x)
            loss_behavior = loss_func(z_pred, z.type(torch.LongTensor).cuda())
            loss_classifier = loss_func(y_pred, y.type(torch.LongTensor).cuda())
            loss = loss_classifier - loss_behavior
            optimizer_classifier.zero_grad()
            optimizer_domain.zero_grad()
            loss.backward()
            optimizer_classifier.step()
            optimizer_domain.step()
    if epoch % 1 == 0:
        with torch.no_grad():
            corrects = torch.zeros(1).cuda()
            fp, tp, fn, tn = 0, 0, 0, 0
            for step, sample in enumerate(test_loader):
                test_x = sample[0].cuda()
                test_y = sample[1].cuda()
                test_output = adversarial_model(test_x)[0]
                pred_y = torch.max(test_output, 1)[1].cuda().data
                corrects += (pred_y == test_y).sum()

                fp += torch.sum((pred_y == 1) & (test_y == 0))
                tp += torch.sum((pred_y == 1) & (test_y == 1))
                
                fn += torch.sum((pred_y == 0) & (test_y == 1))
                tn += torch.sum((pred_y == 0) & (test_y == 0))
            fpr = fp / (fp + tn)
            tpr = tp / (tp + fn)
            fnr = fn / (tp + fn)
            accuracy = int(corrects) /int(mark_test_y) 
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                print('Current Best Acc %.4f, FAR %.4f，FRR %.4f @Epoch %d'%(max_accuracy, fpr, fnr, epoch))
            corrects = torch.zeros(1).cuda()
            for step, sample in enumerate(validate_loader):
                validate_x = sample[0].cuda()
                validate_y = sample[1].cuda()
                validate_output = adversarial_model(validate_x)[0]
                
                pred_y = torch.max(validate_output, 1)[1].cuda().data
                corrects += (pred_y == validate_y).sum()
            validate_accuracy = corrects / int(mark_validate_y) 
            print('Epoch: ', epoch, '| total loss: %.4f' % loss.data.cpu().numpy(),
                  '| classifier loss: %.4f' % loss_classifier.data.cpu().numpy(),
                  '| behavior loss: %.4f' % loss_behavior.data.cpu().numpy(),
                  '| validate accuracy: %.4f' % validate_accuracy,
                  '| test Acc: %.4f' % accuracy,
                  '| test FAR: %.4f' % fpr,
                  '| test FRR: %.4f' % fnr,
                  file=open("log-" + str(ts) + "-log.txt", "a"))
