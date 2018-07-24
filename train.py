# Program to train model for flower recognition
# written by bijan fallah 
# https://www.linkedin.com/in/bijanfallah

'''
'''
#Importing the packages: 
import argparse
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


# command lines for argument parsing:

parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('--data_dir',type=str,help='data directory')
parser.add_argument('--save_dir',type=str,help='saving directory')
parser.add_argument('--arch',type=str,help='torch model architecture')
parser.add_argument('--learning_rate',type=float,help='learning rate')
parser.add_argument('--gpu',type=str,help='gpu or cpu')
parser.add_argument('--top_k',type=int,help='top k predictions')
parser.add_argument('--chpo',type=str,help='checkpoint file')
parser.add_argument('--hidden_units',type=int,help='number of hidden units')
parser.add_argument('--epochs',type=int,help='number of epochs')

arg, _ = parser.parse_known_args() # partial parsing! (https://docs.python.org/2/library/argparse.html)

# functions: 
def validation(model, testloader, criterion):
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:

      
        images, labels = images.to('cuda'), labels.to('cuda')
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy


#load pretraind model  and train it: 
def train_model(data='./flow',arch='vgg19',num_labels=102,hidden=4096,lr=0.001,gpu='gpu',checkpoint='./',epochs=10, print_every=500):
    
    if arg.arch:    
        arch=arg.arch
    if arg.learning_rate:
        lr = arg.learning_rate        
    if arg.gpu:
        gpu = arg.gpu
    if arg.hidden_units:
        hidden = arg.hidden_units
    if arg.chpo: 
        checkpoint = arg.chpo
    if arg.epochs: 
        epochs = arg.epochs
    if arg.save_dir: 
        save_dir = arg.save_dir    
    if arg.data_dir:
        data = arg.data_dir
        
        
#----------------- -----------------------------------------------------    

    if arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch== 'ResNet-101':
        model = models.resnet101(pretrained=True)    
    elif arch== 'ResNet-152':
        model = models.resnet152(pretrained=True)        
    else:
        raise ValueError('model is not amonmg vgg19, vgg16, ResNet-101, ResNet-151: Add in the train.py file manually!')    
 
    for param in model.parameters():
        param.requires_grad = False

    features = list(model.classifier.children())[:-1]   
   
    
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(25088,hidden )),
                            ('relu1', nn.ReLU(True)), 
                            ('dropout1',nn.Dropout(p=.5)),
                            ('fc2', nn.Linear(hidden, hidden)),
                            ('relu2', nn.ReLU(True)),
                            ('dropout2',nn.Dropout(p=.5)),
                            ('fc3', nn.Linear(hidden, num_labels)),
                          
                            ]))
    
    model.classifier = classifier
    print(model)
    data_dir = data
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                             [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                             [0.229, 0.224, 0.225])])


    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=4)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=4)
    import json
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)    
        
    if gpu == 'gpu'    :
        model.to('cuda')
           # optimizer:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=lr)        
    steps = 0
    running_loss = 0
    accuracy_train = 0
    running_accuracy = 0
    for e in range(epochs):
           
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            ps = torch.exp(outputs)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy_train += equality.type(torch.FloatTensor).mean()
            
            if steps % print_every == 0:
                
                model.eval()
                with torch.no_grad():
                    test_loss, accuracy = validation(model, validloader, criterion)
               
                    
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Train Accuracy: {:.3f}".format(accuracy_train/print_every),
                      "Valid Loss: {:.3f}.. ".format(test_loss/len(validloader)),
                      "Valid Accuracy: {:.3f}".format(accuracy/len(validloader)))
            
            
                running_loss = 0
                accuracy_train = 0
                # Training on:
                model.train()
        
    model.class_to_idx = train_dataset.class_to_idx
    checkpoint_dict = {
                'class_to_idx': model.class_to_idx, 
                'optimizer_state': optimizer.state_dict(),    
                'state_dict': model.state_dict(),
                'epochs': 10
            }
            
    torch.save(checkpoint_dict, checkpoint+'model_checkpoint.pth')
    torch.save(model, checkpoint+'model_main_checkpoint.pth')
    print('training finished! and model saved! Enjoy it and do not forget to give feedbacks : https://www.linkedin.com/in/bijanfallah')
    
train_model()
