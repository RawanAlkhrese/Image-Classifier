
import argparse
import numpy as np
import torch 
from torch import nn, optim 
from torchvision import datasets, transforms, models
from PIL import Image
import json

# required arg
parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', action='store',  help='data directory', type=str)
parser.add_argument('--save_dir', default="ImageClassifier", type= str,  action='store', help='storing directory')
parser.add_argument('--arch',  default='vgg19', type= str,  action='store', choices=['vgg19', 'vgg16'] , help='pretrained model the options are vgg19 or vgg16')
parser.add_argument('--learning_rate',  default= 0.0002,  action='store', type=float , help='learning rate')
parser.add_argument('--hidden_units',  default= 100,  action='store', type=int , help='hidden nodes')
parser.add_argument('--epochs',  default= 4,  action='store', type=int , help='number of epochs')
parser.add_argument('--gpu',  action='store_true',  help='move to gpu')

args = parser.parse_args()

#save vlues
data_dir = args.data_dir
arch = args.arch
lr = args.learning_rate
hus = args.hidden_units
epochs = args.epochs
gpu = args.gpu
save_dir = args.save_dir

#directories
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

#transformation
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

#Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader= torch.utils.data.DataLoader(valid_data, batch_size=32)

#Label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

#check the arch
if arch == 'vgg19':
    model = models.vgg19(pretrained=True)
    inputs = model.classifier[0].in_features 
elif arch == 'vgg16':
    model = models.vgg16(pretrained=True)
    inputs = model.classifier[0].in_features 
else:
    raise ValueError('Unexpected network architecture, you have two option: vgg19 or vgg16')
    
# freeze model parameters 
for param in model.parameters():
    param.requires_grad = False

#build classifier
classifier = nn.Sequential( nn.Linear(inputs,hus),
                           nn.ReLU(),
                           nn.Dropout(p=0.2),
                           nn.Linear(hus,102),
                           nn.LogSoftmax(dim=1)
)
#change arch classifer
model.classifier = classifier
#move to gpu
if args.gpu:
    model = model.to('cuda')
#loss function
criterion = nn.NLLLoss()
#optimizer , update only model parameters
optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

steps = 0
running_loss = 0
print_every = 5

for epoch in range(epochs):
    for train_imgs , train_labels in trainloader:
        steps += 1
        #move to gpu
        if args.gpu:
         train_imgs = train_imgs.to('cuda')
         train_labels = train_labels.to('cuda')
        #zero the gradient in each training pass
        optimizer.zero_grad()
        #forward
        train_output = model.forward(train_imgs)
        #cost
        train_loss = criterion(train_output,train_labels)
        #backward
        train_loss.backward()
        #update 
        optimizer.step()
        
        running_loss += train_loss.item()
        
        if steps % print_every == 0:
            # turn the Dropout off 
            model.eval()
            
            valid_loss = 0
            valid_accuracy = 0
            # turn the grads off to speed up the process
            with torch.no_grad():
                for valid_imgs , valid_labels in validloader:
                    if args.gpu:
                        valid_imgs = valid_imgs.to('cuda')
                        valid_labels = valid_labels.to('cuda')
                    valid_output = model.forward(valid_imgs)
                    loss = criterion(valid_output,valid_labels)
                    
                    valid_loss += loss.item()
                    
                    #accuracy 
                    #probabilities
                    ps = torch.exp(valid_output)
                    #the top probab & class label
                    top_p, top_class = ps.topk(1, dim=1)
                    #calculate accuracy
                    equals = top_class == valid_labels.view(*top_class.shape)
                    valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                  f"Valid accuracy: {valid_accuracy/len(validloader):.3f}")
            
            running_loss = 0
            #turn Dropout on for training
            model.train()
            
#save the model
checkpoint = {'classifier':classifier,
              'state_dict':optimizer.state_dict(),
              'class_to_idx':train_data.class_to_idx,
              'epochs' :epochs,
             }
torch.save(checkpoint, save_dir+'/checkpoint.pth')

if args.save_dir:
    print('the model has been sucessfully saved')