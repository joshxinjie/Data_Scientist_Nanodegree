import argparse
import numpy as np
from collections import OrderedDict
from PIL import Image
import json

import torch
from torch import nn
from torchvision import transforms,models

# Setting random seeds
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

def process_argument():
    parser = argparse.ArgumentParser()
    # Default argument
    parser.add_argument("image_path", type=str, help="Path of image to be tested")
    parser.add_argument("checkpoint", type=str, help="Trained model to be loaded")
    parser.add_argument("--top_k", default=3, type=int, help="Return top k predictions")
    parser.add_argument("--category_names", default='', type=str, help="Path of file containing mapping of categories to names")
    parser.add_argument('--gpu', default=False, action='store_true', help='Use GPU processing')
    args = parser.parse_args()
    return args

def load_checkpoint(filename, gpu):
    """
    Loads a trained model from a given saved checkpoint.
    """
    if gpu and torch.cuda.is_available():
        checkpoint = torch.load(filename)
    else:
        # Load GPU model on CPU
        checkpoint = torch.load(filename, \
                                map_location=lambda storage, \
                                loc: storage)
    arch = checkpoint['arch']
    input_size = checkpoint['input_size']
    hidden_units = checkpoint['hidden_units']
    if arch == "alexnet":
        model = models.alexnet(pretrained=True)
    elif arch == "vgg13":
        model = models.vgg13(pretrained=True)
    elif arch == "vgg13_bn":
        model = models.vgg13_bn(pretrained=True)
    elif arch == "resnet34":
        model = models.resnet34(pretrained=True)
    elif arch == "densenet161":
        model = models.densenet161(pretrained=True)
    else:
        raise ValueError('Unexpected network architecture', model)
    
    for param in model.parameters():
        param.requires_grad = False
     
    # There are 102 flower categories
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size, hidden_units)), \
                                            ('relu1', nn.ReLU()), \
                                            ('drop1', nn.Dropout(p=0.33)), \
                                            ('fc2', nn.Linear(hidden_units, 102)), \
                                            ('output', nn.LogSoftmax(dim=1))]))
    classifier_models = ["alexnet", "vgg13", "vgg13_bn", "densenet161"]
    if arch in classifier_models:
        model.classifier = classifier
    else:
        # Doesn't work for resnet. resolve this
        model.fc = classifier
    
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    epochs = checkpoint['epochs']
    optimizer = checkpoint['optimizer']
    criterion = checkpoint['criterion']
    
    return model, criterion, optimizer, epochs

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    transformations = transforms.Compose([transforms.Resize(256), \
                                         transforms.CenterCrop(224), \
                                         transforms.ToTensor(), \
                                         transforms.Normalize([0.485, 0.456, 0.406], \
                                                             [0.229, 0.224, 0.225])])
    
    pil_image = transformations(pil_image)
    return pil_image

def predict(image_path, model, gpu, category_names, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    if gpu and torch.cuda.is_available():
        model.to('cuda')
    else:
        model.to('cpu')
    processed_image_torch = process_image(image_path)
    # Batch dimension is missing. Adds a dimension of size one at the specified position 0.
    processed_image_torch = processed_image_torch.unsqueeze_(0)
    # If use gpu and gpu is available
    if gpu and torch.cuda.is_available():
        image = processed_image_torch.to('cuda')
    else:
        image = processed_image_torch.to('cpu')
    output = model.forward(image)
    # Get top k largest values in a tensor and their indices
    topk_log_probs, topk_classes = output.topk(topk)
    # Convert to numpy probabilities
    if gpu:
        topk_log_probs = topk_log_probs.data.cpu().numpy()[0]
        topk_classes = topk_classes.data.cpu().numpy()[0]
    else:
        topk_log_probs = topk_log_probs.data.numpy()[0]
        topk_classes = topk_classes.data.numpy()[0]
    # Getting back softmax probabilities from log softmax probabilities
    topk_probs = np.exp(topk_log_probs)
    # model.class_to_idx contains {'actual class labels': training indices}
    # invert this to {'training indices': actual class labels}
    idx_to_class = {x:y for y,x in model.class_to_idx.items()}
    topk_classes = [idx_to_class[i] for i in topk_classes]
    
    # Printing Results
    # If mapping of categories to names is provided
    if len(category_names) > 0:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        topk_flower_names = [cat_to_name[x] for x in topk_classes]
        for i in range(topk):
            print("{}, {:.4f}".format(topk_flower_names[i], topk_probs[i]))
    else:
        # No mapping of categories to names is provided
        for i in range(topk):
            print("Class {}, {:.4f}".format(topk_classes[i], topk_probs[i]))

if __name__ == '__main__':
    args = process_argument()
    model, criterion, optimizer, epochs = load_checkpoint(args.checkpoint, args.gpu)
    predict(args.image_path, model, args.gpu, args.category_names, args.top_k)
    
    # Test cases
    # python predict.py --top_k 5 --category_names ../aipnd-project/cat_to_name.json --gpu ../aipnd-project/flowers/test/10/image_07090.jpg checkpoint.pth
    # python predict.py --category_names ../aipnd-project/cat_to_name.json ../aipnd-project/flowers/test/10/image_07090.jpg checkpoint.pth
    # python predict.py --top_k 5 --gpu ../aipnd-project/flowers/test/10/image_07090.jpg checkpoint.pth