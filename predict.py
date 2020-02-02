import argparse
import numpy as np
import torch 
from torch import nn, optim 
from torchvision import datasets, transforms, models
from PIL import Image
import json

#Label mapping  



parser = argparse.ArgumentParser()

parser.add_argument('--img_path', action='store',  help='path_to_image', type=str)
parser.add_argument('--checkpoint', action='store',  help='checkpoint model to be used', type=str)
parser.add_argument('--top_k', type= int,  default=5, action='store', help='number of most likely classes')
parser.add_argument('--category_names',  action='store', default= 'cat_to_name' , help='path of category names', type=str)
parser.add_argument('--gpu',  action='store_true',  help='move to gpu')

args= parser.parse_args()

img= args.img_path
top_k= args.top_k
category_names= args.category_names
gpu= args.gpu
checkpoint= args.checkpoint

if args.category_names:
    with open('cat_to_name.json', 'r') as f:
      cat_to_name = json.load(f)
    
#process image 
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #Process a PIL image for use in a PyTorch model 
    img_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    img = img_transforms(Image.open(image))
    return img

def predict(image_path, model, top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    #Implement the code to predict the class from an image file
    with torch.no_grad():
        #move to cpu
        if args.gpu:
            model.cuda()
        else:
            model.cpu()
        model.eval()
        img = process_image(image_path)
        # add dimention , which is the size of input (in our case only 1 image)
        img= img.unsqueeze(0)
            if args.gpu:
                img = img.to('cuda')
            else:
                img = img.type(torch.FloatTensor)
        output = model.forward(img)
        #probabilities
        ps = torch.exp(output)
        #top 5 probabilities & classes
        top_p, top_class = ps.topk(top_k)
        #invert class_to_idx so we get a mapping from index to class 
        idx_to_class = {val:key for key, val in model.class_to_idx.items()}
        if args.gpu:
            classes = [idx_to_class[c] for c in top_class.cpu().numpy()[0]]
            top_p = top_p.cpu().numpy()[0]
        else:
            classes = [idx_to_class[c] for c in top_class.numpy()[0]]
            top_p= top_p.numpy()[0]
        #get the top 5 flowers name using cat_to_name.json file which we already loaded
        flowers_names = [cat_to_name[idx_to_class[lab]] for lab in top_class.numpy()[0]]
        return top_p , classes, flowers_names


#load model 
def loading_checkpoint(filename):
    checkpoint = torch.load(filename)
    
    model = models.vgg19(pretrained=True)
    model.classifier= checkpoint['classifier']
    model.state_dict = checkpoint['state_dict']
    model.class_to_idx = checkpoint['class_to_idx']
    
    for param in model.parameters(): 
        param.requires_grad = False 
    return model 

model = loading_checkpoint(checkpoint)

probs , classes , names = predict(img,model,top_k)
flower_idex = img.split('/')[2]
flower_name = cat_to_name[flower_idex]
      
if args.top_k:
    print('top K most likely classes: ')
    print(classes)
    print('and their probabilities:')
    print(probs)

print('flower name: ' + flower_name)
print( 'with probabilty: '+ str(probs[0]))