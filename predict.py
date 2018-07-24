# Program to predict the flower name 
# written by bijan fallah 

import argparse
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import json
parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('--input',type=str,help='input image')
parser.add_argument('--gpu',type=str,help='gpu or cpu')
parser.add_argument('--top_k',type=int,help='top k predictions')
parser.add_argument('--checkpoint',type=str,help='checkpoint file')
parser.add_argument('--category_names',type=str,help='category names')

arg, _ = parser.parse_known_args() # partial parsing! (https://docs.python.org/2/library/argparse.html)


def process_image(image):
    from PIL import Image
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    img_load =  transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor()])
                                    
    img = Image.open(image)
    img = img_load(img).float()
    img = np.array(img)                                
    mean = np.array([0.485, 0.456, 0.406])
    sd = np.array([0.229, 0.224, 0.225])
    img = (np.transpose(img, (1, 2, 0)) - mean)/sd    
    img = np.transpose(img, (2, 0, 1))                                
                                    
    return img

def predict(inputs='flowers/test/1/image_06743.jpg', top_k=3, check='./', gpu='gpu',category_names='cat_to_name.json'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if arg.input:    
        inputs=arg.input
    if arg.top_k:
        top_k = arg.top_k       
    if arg.gpu:
        gpu = arg.gpu
    if arg.checkpoint: 
        check = arg.checkpoint
    if arg.category_names:
        category_names = arg.category_names
    
    
    checkpoint = torch.load(check+'model_checkpoint.pth')
    model_new = torch.load(check+'model_main_checkpoint.pth')

    num_labels = len(checkpoint['class_to_idx'])

    model_new.load_state_dict(checkpoint['state_dict'])
    model_new.class_to_idx = checkpoint['class_to_idx']
    model_new.epochs = checkpoint['epochs']
    def getKeysByValues(dictOfElements, listOfValues):
        listOfKeys = list()
        listOfItems = dictOfElements.items()
        for item  in listOfItems:
            if item[1] in listOfValues:
                listOfKeys.append(item[0])
        return  listOfKeys 
    
    
 

    
    img =  process_image(inputs)
    img = Variable(torch.FloatTensor(img), requires_grad=True)
    img = img.unsqueeze(0)
    output = model_new(img.to('cuda'))
    pred = output.topk(top_k)
    prob = torch.nn.functional.softmax(pred[0].data, dim=1).cpu().numpy()[0]
    lab = pred[1].data.cpu().numpy()[0]
    label = checkpoint['class_to_idx']    
    keys = getKeysByValues(label, lab )
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    print([cat_to_name[x] for x in keys], np.array(prob,dtype=np.float32) )


predict()
#python predict.py /path/to/image checkpoint 
