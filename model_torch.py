import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F
from PIL import Image
from torchvision.io import read_image
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF
from scipy.stats import kendalltau

## UTILS

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_onnx_graph(model, batch_shape = (1, 3, 300, 300), input_names = ['Image'], output_names = ['Matrix']):
    torch.onnx.export(model, torch.rand(1, 3, 300, 300), 'rnn.onnx', input_names=input_names, output_names=output_names)
    print('Saved Onnx!!!')


def convert_pred_to_label(pred, plot_img = True):
    print('Converting prediction to label...')
    if len(pred.shape) == 3:
        pred = pred[0, :, :]
    for i in range(len(pred[0])):
        pred[i] = pred[i]==torch.max(pred[i]).float()
    if plot_img:
        x = pred.detach().numpy()
        plt.figure(figsize=(9, 9))
        plt.imshow(x)
        plt.show()
        return
    else:
        return pred.detach().numpy()

def calculate_metrics(y_true, y_pred):
    tau = kendalltau(y_true, y_pred)
    hs = hamming_loss(y_pred, y_true)
    return tau, hs

def model_predict(model, img, to_label=False):
    img = Image.open(img)
    img = TF.to_tensor(img)
    img.unsqueeze_(0)
    pred = model(img)
    if to_label:return convert_pred_to_label(pred, False)
    return pred.detach()

def check_model(summary_ = True):
    model = Model()
    print(f'Total Number parameters in Model : {count_parameters(model)}')
    pred = model(torch.rand(1, 3, 300, 300))
    convert_pred_to_label(pred)
    summary(model, (3, 300, 300))

def hamming_loss(prediction, target):
    sim = 0
    for batch in range(len(prediction)):
        for img in range(len(prediction[batch])):
            for row in range(len(prediction[batch][img])):
                if prediction[batch][img][row] == target[batch][img][row]:
                    sim += 1
    return sim / 36
            


# Redundant Code

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
    
    
class Alexnet(nn.Module):
    def __init__(self):
        super(Alexnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=21, kernel_size=6, stride=2)
        self.bn1 = nn.BatchNorm2d(21)
        self.mp1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=21, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.mp2 = nn.MaxPool2d(kernel_size=3, stride=1)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=43, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(43)
        
        self.conv4 = nn.Conv2d(in_channels=43, out_channels=43, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(43)
        
        self.conv5 = nn.Conv2d(in_channels=43, out_channels=21, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(21)
        self.mp3 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.fl = nn.Flatten()
        
        self.fc6 = nn.Linear(in_features=336, out_features=72)
    
            
    def conv_1(self, inputs):
        conv = self.conv1(inputs)
        bn = self.bn1(conv)
        mp = self.mp1(F.relu(bn))
        return mp

    def conv_2(self, inputs):
        conv = self.conv2(inputs)
        bn = self.bn2(conv)
        mp = self.mp2(F.relu(bn))
        return mp
    
    def conv_3(self, inputs):
        conv = self.conv3(inputs)
        bn = self.bn3(conv)
        return F.relu(bn)
    
    def conv_4(self, inputs):
        conv = self.conv4(inputs)
        bn = self.bn4(conv)
        return F.relu(bn)
    
    def conv_5(self, inputs):
        conv = self.conv5(inputs)
        bn = self.bn5(conv)
        mp = self.mp3(F.relu(bn))
        return mp
    
    def fullyConnected_6(self, inputs):
        flat = self.fl(inputs)
        fullyConnected = self.fc6(flat)
        return F.relu(fullyConnected)

    def forward(self, inputs):
        l1 = self.conv_1(inputs)
        l2 = self.conv_2(l1)
        l3 = self.conv_3(l2)
        l4 = self.conv_4(l3)
        l5 = self.conv_5(l4)
        l6 = self.fullyConnected_6(l5)
        return l6

class SinkhornLayer(nn.Module):
    def __init__(self, n_iters=21, temperature=0.01, **kwargs):
        super(SinkhornLayer, self).__init__(**kwargs)
        self.supports_masking = False
        self.n_iters = n_iters
        self.temperature = torch.tensor(temperature)

    def forward(self, input_tensor, mask=None):
        n = input_tensor.shape[1]
        log_alpha = input_tensor.view(-1, n//36, n//36)
        log_alpha /= self.temperature

        for _ in range(self.n_iters):
            log_alpha -= torch.log(torch.sum(torch.exp(log_alpha), dim=2)).unsqueeze(2)
            log_alpha -= torch.log(torch.sum(torch.exp(log_alpha), dim=1)).unsqueeze(1)

        return torch.exp(log_alpha)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cnn = Alexnet()
        self.fc7 = nn.Linear(in_features = 2592, out_features = 2592)
        self.fc8 = nn.Linear(in_features = 2592, out_features = 1296)

    def forward(self, img):
        patches = []
        for i in range(6):
            for j in range(6):
                patch = img[:, :, i * 50 : (i * 50) + 50, j * 50: (j * 50) + 50]
                patches.append(self.cnn(patch))
        cat = torch.cat(patches, axis = -1)
        cat_1 = self.fc7(cat)
        cat_2 = self.fc8(cat_1)
        sk = SinkhornLayer()(cat_2)
        return sk