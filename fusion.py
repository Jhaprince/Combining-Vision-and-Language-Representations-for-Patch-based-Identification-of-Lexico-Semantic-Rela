# Import statements:

# Importing the significant libraries
import pandas as pd
import pickle as pkl
from pandas import ExcelWriter
import numpy as np, pandas as pd
from numpy.random import seed
from pickle import dump

from pathlib import Path

# Importing functions from other files of this program
from main_utils import load_embeddings, train_model
from func_train_test_image import func_train_test_image
from func_train_test_text import func_train_test_text
#from nn_models import NN_Baseline
#from attention_fusion_network import visual_textual_MLP_model_glo_att

import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score,precision_score, recall_score,classification_report
import matplotlib.pyplot as plt
import os
from central_net import CentralNet
import torch 
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F 
from torch.utils.data import TensorDataset, DataLoader
import tensorflow as tf
from tensorflow import keras
import sys
from pytorchtools import EarlyStopping


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)






# Initializing all the important variables
file_paths = {'fig':'E:/CentralNet/results/k=1/Root9/loss_curves',
              'glove_file':    '/home/prince/prince/CentralNet/glove/glove.6B.300d.txt',
              'wnvec_file':    '/home/prince/prince/CentralNet/data/image/pre-processData-new.pkl',
              'temp_file':     '/home/prince/prince/CentralNet/data/text_word2vec.txt'}



# initialization of all hyper-parameters (including flags)
k =0
# Patch size
hps = {'concat':        False,
       'diff':          False,
       'cosine':        False,
       'neighbors':     False,
       'diffPatch':     True,
       'attNeigh':      True,
       'concatNeigh':   True,
       'patchAttCNN':   False,
       'diffPatchAtt':  True,
       'setZeros':      True,  # always keep the value as True
       'cuboid':        False,
       'glove_dim':      300}





# possible datasets at our disposal: #'Rumen','Root9','Bless','Cogalex','Weeds','Root9_Bal','Bless_Bal','Cogalex_Bal','Weeds_Bal','Root9+Bless+Weeds','Root9+Bless+Rumen'

datasets   = ['Rumen'] # List of datasets we would process together

name_data="Rumen"


# load the glove embeddings
#embeddings = load_embeddings(file_paths['glove_file'], hps['glove_dim'])

# Load the word based data
#with open(file_paths['wnvec_file'], 'rb') as f: # File containing list of words and neighbors
#    wnVectors = pkl.load(f)
#print('wnVectors loaded')

#data_text= func_train_test_text(wnVectors, name_data, embeddings, k, hps, file_paths)
text_feat=os.path.join("/home/prince/prince/CentralNet/data",name_data,name_data+"_txt_"+str(k)+".pkl")
#dump(data_text, open(text_feat,'wb'))
with open(text_feat,"rb") as f:
    data_text=pkl.load(f)
    
print("text data loaded")


#data_image= func_train_test_image(wnVectors, name_data, k, hps, file_paths)
image_feat=os.path.join("/home/prince/prince/CLIP/CentralNet-img/data",name_data,name_data+"_img_"+str(k)+".pkl")
#dump(data_image, open(image_feat,'wb'))

with open(image_feat,"rb") as f:
    data_image=pkl.load(f)


print("image data loaded")

#print('hereFin')





    
class Lexical(torch.utils.data.Dataset):
    def __init__(self,data_txt,data_img,flag,name):
        self.data_txt = data_txt
        self.data_img = data_img
        self.flag = flag
        self.name=name
        
    def __len__(self):
        
        if(self.flag == "train"):
            return len(self.data_txt[self.name]["y_train"])
        if(self.flag == "val"):
            return len(self.data_txt[self.name]["y_valid"])
        if(self.flag == "test"):
            return len(self.data_txt[self.name]["y_test"])
    
    def __getitem__(self,idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        typ = self.name
            
        if(self.flag == "train"):    
            label = torch.tensor(self.data_txt[typ]["y_train"][idx]).long().to(device)
            txt_feat = torch.tensor(self.data_txt[typ]["x_train"][idx]).float().to(device)
            img_feat = torch.tensor(self.data_img[typ]["x_train"][idx]).float().to(device)
            
                    
            
        if(self.flag == "val"):    
            label = torch.tensor(self.data_txt[typ]["y_valid"][idx]).long().to(device)
            txt_feat = torch.tensor(self.data_txt[typ]["x_valid"][idx]).float().to(device)
            img_feat = torch.tensor(self.data_img[typ]["x_valid"][idx]).float().to(device)
            
            
                    
            
        if(self.flag == "test"):    
            label = torch.tensor(self.data_txt[typ]["y_test"][idx]).long().to(device)
            txt_feat = torch.tensor(self.data_txt[typ]["x_test"][idx]).float().to(device)
            img_feat = torch.tensor(self.data_img[typ]["x_test"][idx]).float().to(device)
            
         
        
        
        sample = {
            "txt_feat": txt_feat,
            "img_feat":img_feat,
            "label":label
        }
        
        return sample
        
        
        

#Coord_train = Lexical(data_text,data_image,"train","Coord-Random")
#dataloader_train = DataLoader(Coord_train,batch_size=32, shuffle=False, num_workers=0)


Hyper_train = Lexical(data_text,data_image,"train","Hyper-Random")
dataloader_train = DataLoader(Hyper_train,batch_size=32, shuffle=False, num_workers=0)

#Coord_val =Lexical(data_text,data_image,"val","Coord-Random")
#dataloader_val = DataLoader(Coord_val, batch_size=32, shuffle=False, num_workers=0)

Hyper_val =Lexical(data_text,data_image,"val","Hyper-Random")
dataloader_val = DataLoader(Hyper_val, batch_size=32, shuffle=False, num_workers=0)

#Coord_test = Lexical(data_text,data_image,"test","Coord-Random")
#dataloader_test = DataLoader(Coord_test,batch_size=32,shuffle=False, num_workers=0)


Hyper_test = Lexical(data_text,data_image,"test","Hyper-Random")
dataloader_test = DataLoader(Hyper_test,batch_size=32,shuffle=False, num_workers=0)









class text_model(nn.Module):
    def __init__(self,k,n_out):
        super(text_model,self).__init__()
        self.txt_512 = nn.Linear((2*(k+1)*300)+((k+1)**2),512)
        self.txt_256 = nn.Linear(512,256)
        self.txt_out = nn.Linear(256,n_out)
    def forward(self,txt,img):
        
        txt_a = self.txt_512(txt)
        txt_b = self.txt_256(txt_a)
        txt_out = self.txt_out(txt_b)
        
        return txt_a,txt_b,txt_out

    
    
    
class image_model(nn.Module):
    def __init__(self,k,n_out):
        super(image_model,self).__init__()
        self.img_512 = nn.Linear((2*(k+1)*512)+((k+1)**2),512)
        self.img_256 = nn.Linear(512,256)
        self.img_out = nn.Linear(256,n_out)
    def forward(self,txt,img):
    
        img_a = self.img_512(img)
        img_b = self.img_256(img_a)
        img_out = self.img_out(img_b)
        
        return img_a,img_b,img_out
        
        
        
class Attention_Fusion(nn.Module):
    def __init__(self,k,n_out):
        super(Attention_Fusion,self).__init__()
        self.img_512 = nn.Linear((2*(k+1)*512)+((k+1)**2),512)
        self.txt_512 = nn.Linear((2*(k+1)*300)+((k+1)**2),512)
        
        self.sigmoid =nn.Sigmoid()
        self.tanh = nn.ReLU()
        
        self.prob_img = nn.Linear(1024,1)
        self.prob_txt = nn.Linear(1024,1)
        
        self.concat_512 = nn.Linear(1024,512)
        self.concat_256 = nn.Linear(512,256)
        
        self.out = nn.Linear(256,2)
        
    def forward(self, txt, img):
    
        txt_512 = self.tanh(self.txt_512(txt))
        img_512 = self.tanh(self.img_512(img))
        
        concat = torch.cat((txt_512,img_512),dim=1)
        
        prob_txt = self.sigmoid(self.prob_txt(concat))
        prob_img = self.sigmoid(self.prob_img(concat))
        
        concat = self.tanh(torch.cat((prob_txt*txt_512,prob_img*img_512),dim=1))
        
        concat_512 = self.tanh(self.concat_512(concat))
        concat_256 = self.tanh(self.concat_256(concat_512))
        
        return self.out(concat_256)
        
        
        
    

    
class CentralNet(nn.Module):
    def __init__(self,k,n_out):
        super(CentralNet,self).__init__()
        self.image_model = image_model(k,n_out)
        self.text_model = text_model(k,n_out)
        
        self.img_txt_256 = nn.Linear(512,256)
        self.img_txt_out = nn.Linear(256,n_out)
        
        #self.tanh = nn.ReLU()
        
        
        self.alphas_t = nn.ParameterList([torch.nn.Parameter(torch.rand(1)) for i in range(4)])
        self.alphas_v = nn.ParameterList([torch.nn.Parameter(torch.rand(1)) for i in range(4)])
        self.alphas_c = nn.ParameterList([torch.nn.Parameter(torch.rand(1)) for i in range(4)])
        
        
    def forward(self, txt, img):
        self.image_model.load_state_dict(torch.load("/home/prince/prince/clip_glove/path_to_saved_files/Image_ModelCkpt/EMNLP_MCHarm_GLAREAll_COVTrain_POLEval/checkpoint_EMNLP_MCHarm_GLAREAll_COVTrain_POLEval.pt"))
        
        for p in self.image_model.parameters():
            p.requires_grad=False
            
            
        self.text_model.load_state_dict(torch.load("/home/prince/prince/clip_glove/path_to_saved_files/Text_ModelCkpt/EMNLP_MCHarm_GLAREAll_COVTrain_POLEval/checkpoint_EMNLP_MCHarm_GLAREAll_COVTrain_POLEval.pt"))
        
        for p in self.text_model.parameters():
            p.requires_grad=False
            
            
            
        txt_a, txt_b , txt_out = self.text_model(txt,img)
        img_a, img_b, img_out = self.image_model(txt,img)
        
        sz = img.shape[0]
        
        
        central_rep = torch.zeros((sz,512)).to(device)
        
        xsum_a = txt_a*self.alphas_t[0].expand_as(txt_a) + central_rep*self.alphas_c[0].expand_as(central_rep) + img_a*self.alphas_v[0].expand_as(img_a)
        
        central_rep = self.img_txt_256(xsum_a)
        
        xsum_b = txt_b*self.alphas_t[1].expand_as(txt_b) + central_rep*self.alphas_c[1].expand_as(central_rep) + img_b*self.alphas_v[1].expand_as(img_b)
        
        central_rep = self.img_txt_out(xsum_b)
        #central_rep = self.img_txt_out(central_rep)
        
        
        xsum_out = txt_out*self.alphas_t[2].expand_as(txt_out) + central_rep*self.alphas_c[2].expand_as(central_rep) + img_out*self.alphas_v[2].expand_as(img_out)
        
        
        
        
        return xsum_out
        #return  central_rep
        #return txt_out,img_out,central_rep
        #return txt_out,img_out,xsum_out

        
class MultiTask(nn.Module):
    def  __init__(self,k,n_out):
        super(MultiTask,self).__init__()
        self.img_4096 = nn.Linear(2*(k+1)*4096+((k+1)**2),4096)
        self.img_1024 = nn.Linear(4096,1024)
        self.img_512 = nn.Linear(1024,512)
        self.img_256 = nn.Linear(512,256)
        self.img_out = nn.Linear(256,n_out)
        
        self.txt_512 = nn.Linear((2*(k+1)*300)+((k+1)**2),512)
        self.txt_256 = nn.Linear(512,256)
        self.txt_out = nn.Linear(256,n_out)
        
        self.tanh = nn.Tanh()
        
        
        self.alphas_t = nn.ParameterList([torch.nn.Parameter(torch.rand(1)) for i in range(1)])
        self.alphas_i = nn.ParameterList([torch.nn.Parameter(torch.rand(1)) for i in range(1)])
        
    def forward(self, txt, img):
        img_a = self.tanh(self.img_4096(img))
        img_b = self.tanh(self.img_1024(img_a))
        img_c = self.tanh(self.img_512(img_b))
        img_d = self.tanh(self.img_256(img_c))
        img_out = self.img_out(img_d)
        
        txt_a = self.tanh(self.txt_512(txt))
        txt_b = self.tanh(self.txt_256(txt_a))
        
        xsum = txt_b*self.alphas_t[0].expand_as(txt_b) + img_d*self.alphas_i[0].expand_as(img_d)
        
        txt_out = self.txt_out(xsum)
        
        
        return img_out, txt_out
        
        
    
        
output_size = 2

exp_path = "ACMMM2022"


lr=0.00001
# criterion = nn.BCELoss() #Binary case
criterion = nn.CrossEntropyLoss()
# # ------------Fresh training------------
#model = text_model(k,output_size).to(device)
#model = image_model(k,output_size).to(device)
model = CentralNet(k,output_size).to(device)
#model = Attention_Fusion(k,output_size).to(device)
#model = MultiTask(k,output_size).to(device)
#model.to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)






def train_model(model, patience, n_epochs):
    epochs = n_epochs
#     clip = 5

    train_acc_list=[]
    val_acc_list=[]
    train_loss_list=[]
    val_loss_list=[]
    
        # initialize the experiment path
    Path(exp_path).mkdir(parents=True, exist_ok=True)
    # initialize early_stopping object
    chk_file = os.path.join(exp_path, 'final'.pt')
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=chk_file)
    
    
    


    model.train()
    for i in range(epochs):
#         total_acc_train = 0
        total_loss_train = 0
        total_train = 0
        correct_train = 0

        for data in dataloader_train:
             
            img_feat = data['img_feat'].to(device)
            txt_feat = data['txt_feat'].to(device)

            label_train = data['label'].to(device)

            model.zero_grad()
            #txt_a,txt_b,output = model(txt_feat, img_feat)
            #img,output = model(txt_feat, img_feat)
            output = model(txt_feat, img_feat)
#             print(output.shape)
#             output = model(img_feat_vgg, txt_feat_trans)

            #print("output train",output.shape)

            #loss = criterion(output, label_train)+criterion(txt_a, label_train)+criterion(txt_b, label_train)
            loss = criterion(output, label_train)
            #loss = 0.1*criterion(img, label_train)+criterion(output, label_train)
            
#             print(loss)
            loss.backward()
#             nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            with torch.no_grad():
                _, predicted_train = torch.max(output.data, 1)
                total_train += label_train.size(0)
                correct_train += (predicted_train == label_train).sum().item()
#                 out_val = (output.squeeze()>0.5).float()
#                 out_final = ((out_val == 1).nonzero(as_tuple=True)[0])
#                 print()
#                 acc = torch.abs(output.squeeze() - label.float()).view(-1)
#                 acc = (1. - acc.sum() / acc.size()[0])
#                 total_acc_train += acc
                total_loss_train += loss.item()

        
        train_acc = 100 * correct_train / total_train
        train_loss = total_loss_train/total_train
        model.eval()
#         total_acc_val = 0
        total_loss_val = 0
        total_val = 0
        correct_val = 0

        with torch.no_grad():
            for data in dataloader_val:                
               
                img_feat = data['img_feat'].to(device)                
                txt_feat = data['txt_feat'].to(device)

                

                label_val = data['label'].to(device)

                model.zero_grad()
                
                #txt_a,txt_b,output = model(txt_feat, img_feat)
                output = model(txt_feat,img_feat)
                #img,output = model(txt_feat,img_feat)
#                 output = model(img_feat_vgg, txt_feat_trans)
                
                #print("output val",output.shape)
        
                #val_loss = criterion(output, label_val)+criterion(txt_a, label_val)+criterion(txt_b, label_val)
                val_loss = criterion(output, label_val)
                #val_loss =0.1*criterion(img, label_val) + criterion(output, label_val)
                
                _, predicted_val = torch.max(output.data, 1)
                total_val += label_val.size(0)
                correct_val += (predicted_val == label_val).sum().item()                
                total_loss_val += val_loss.item()
        #print("Saving model...") 
        #torch.save(model.state_dict(), os.path.join(exp_path, "final.pt"))

        val_acc = 100 * correct_val / total_val
        val_loss = total_loss_val/total_val

        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
            
        print(f'Epoch {i+1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}')
        model.train()
        torch.cuda.empty_cache()
        
    # load the last checkpoint with the best model
#     model.load_state_dict(torch.load('checkpoint_1.pt'))
    
    return  model, train_acc_list, val_acc_list, train_loss_list, val_loss_list, i
        


def test_model(model):
    model.eval()
    total_test = 0
    correct_test =0
    total_acc_test = 0
    total_loss_test = 0
    outputs = []
    test_labels=[]
    with torch.no_grad():
        for data in dataloader_test:

            img_feat = data['img_feat'].to(device)
            txt_feat = data['txt_feat'].to(device)            

            label_test = data['label'].to(device)
            
#             out = model(img_feat_vgg, txt_feat_trans)        

            #_,_,out = model(txt_feat, img_feat) 
            #_,out = model(txt_feat, img_feat) 
            out = model(txt_feat, img_feat)

            outputs += list(out.cpu().data.numpy())
            loss = criterion(out, label_test)
            
            _, predicted_test = torch.max(out.data, 1)
            total_test += label_test.size(0)
            correct_test += (predicted_test == label_test).sum().item()
#                 out_val = (output.squeeze()>0.5).float()
#                 out_final = ((out_val == 1).nonzero(as_tuple=True)[0])
#                 print()
#                 acc = torch.abs(output.squeeze() - label.float()).view(-1)
#                 acc = (1. - acc.sum() / acc.size()[0])
#                 total_acc_train += acc
            total_loss_test += loss.item()
            
            
#     #         print(label.float())
#             acc = torch.abs(out.squeeze() - label.float()).view(-1)
#     #         print((acc.sum() / acc.size()[0]))
#             acc = (1. - acc.sum() / acc.size()[0])
#     #         print(acc)
#             total_acc_test += acc
#             total_loss_test += loss.item()

    
    acc_test = 100 * correct_test / total_test
    loss_test = total_loss_test/total_test   
    
    print(f'acc: {acc_test:.4f} loss: {loss_test:.4f}')
    return outputs



n_epochs = 200
# early stopping patience; how long to wait after last time validation loss improved.
patience = 10
model, train_acc_list, val_acc_list, train_loss_list, val_loss_list, epoc_num = train_model(model, patience, n_epochs)


#chk_file = os.path.join(exp_path, 'final'.pt')
#model.load_state_dict(torch.load(chk_file))
    
outputs = test_model(model)

# Multiclass setting - Harmful
y_pred=[]
for i in outputs:
#     print(np.argmax(i))
    y_pred.append(np.argmax(i))
# # np.argmax(outputs[:])
# outputs

# # Multiclass setting
test_labels=[]

for i in range(len(data_text["Hyper-Random"]["y_test"])):
    #test_labels.append(data_text["Coord-Random"]["y_test"][i])
    test_labels.append(data_text["Hyper-Random"]["y_test"][i])

# In[ ]:


def calculate_mmae(expected, predicted, classes):
    NUM_CLASSES = len(classes)
    count_dict = {}
    dist_dict = {}
    for i in range(NUM_CLASSES):
        count_dict[i] = 0
        dist_dict[i] = 0.0
    for i in range(len(expected)):
        dist_dict[expected[i]] += abs(expected[i] - predicted[i])
        count_dict[expected[i]] += 1
    overall = 0.0
    for claz in range(NUM_CLASSES): 
        class_dist =  1.0 * dist_dict[claz] / count_dict[claz] 
        overall += class_dist
    overall /= NUM_CLASSES
#     return overall[0]
    return overall


# In[ ]:


rec = np.round(recall_score(test_labels, y_pred, average="weighted"),4)
prec = np.round(precision_score(test_labels, y_pred, average="weighted"),4)
f1 = np.round(f1_score(test_labels, y_pred, average="weighted"),4)
# hl = np.round(hamming_loss(test_labels, y_pred),4)
acc = np.round(accuracy_score(test_labels, y_pred),4)
#mmae = np.round(calculate_mmae(test_labels, y_pred, [0,1]),4)
#mae = np.round(mean_absolute_error(test_labels, y_pred),4)
# print("recall_score\t: ",rec)
# print("precision_score\t: ",prec)
# print("f1_score\t: ",f1)
# print("hamming_loss\t: ",hl)
# print("accuracy_score\t: ",f1)
print(classification_report(test_labels, y_pred))


# In[ ]:


print("Acc, F1, Rec, Prec, MAE, MMAE")
print(acc, f1, rec, prec)










