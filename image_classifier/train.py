import argparse
import numpy as np
from collections import OrderedDict

import torch
from torch import nn
from torch import optim
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader

# Setting random seed
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

def process_argument():
    """
    Retrieves and parse the command line arguments.
    """
    parser = argparse.ArgumentParser()
    # Default argument
    parser.add_argument("data_dir", type=str, help="Directory of data")
    # Optional arguments
    parser.add_argument("--save_dir", default='', type=str, help="Directory to save checkpoints")
    parser.add_argument("--arch", default='densenet121', type=str, help="Pick a network architecture. Options: alexnet, vgg13, vgg13_bn, resnet34, densenet161", choices=["alexnet", "vgg13", "vgg13_bn", "resnet34", "densenet161"])
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Set learning rate' )
    parser.add_argument('--hidden_units', default=256, type=int, help='Set number of nodes in each hidden layer')
    parser.add_argument('--epochs', default=3, type=int, help='Set number of training epochs')
    parser.add_argument('--gpu', default=False, action='store_true', help='Use GPU processing')
    args = parser.parse_args()
    
    # For checking purposes
    '''
    print(args.data_dir)
    print(args.arch)
    print(args.learning_rate)
    print(args.hidden_units)
    print(args.epochs)
    if args.gpu:
        print("Using GPU")
    '''
    return args

def create_dataloaders(data_dir):
    """
    Creates the data-loaders for the training, validation and test datasets.
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_data_transforms = transforms.Compose([transforms.RandomHorizontalFlip(), \
                                                transforms.RandomRotation(45), \
                                                transforms.RandomResizedCrop(224),\
                                                transforms.ToTensor(), \
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    valid_test_data_transforms = transforms.Compose([transforms.Resize(255), \
                                                    transforms.CenterCrop(224), \
                                                    transforms.ToTensor(), \
                                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_image_datasets = datasets.ImageFolder(train_dir, transform=train_data_transforms)
    valid_image_datasets = datasets.ImageFolder(valid_dir, transform=valid_test_data_transforms)
    test_image_datasets = datasets.ImageFolder(test_dir, transform=valid_test_data_transforms)

    train_dataloaders = DataLoader(train_image_datasets, batch_size=32, shuffle=True)
    valid_dataloaders = DataLoader(valid_image_datasets, batch_size=32, shuffle=True)
    test_dataloaders = DataLoader(test_image_datasets, batch_size=32, shuffle=True)

    class_to_idx = train_image_datasets.class_to_idx
    
    return train_dataloaders, valid_dataloaders, test_dataloaders, class_to_idx

def create_model(model, hidden_units, learning_rate):
    """
    Creates a transfer-learning model to be used for the image classification task. 
    Downloads the specified pre-trained network architecture and freezes its parameters.
    It also creates a new untrained classifier for the model, initializes the loss
    function and the optimizer to be used.
    """
    model_name = model
    if model == "alexnet":
        model = models.alexnet(pretrained=True)
        input_size = model.classifier[1].in_features
    elif model == "vgg13":
        model = models.vgg13(pretrained=True)
        input_size = model.classifier[0].in_features
    elif model == "vgg13_bn":
        model = models.vgg13_bn(pretrained=True)
        input_size = model.classifier[0].in_features
    elif model == "resnet34":
        model = models.resnet34(pretrained=True)
        input_size = model.fc.in_features
    elif model == "densenet161":
        model = models.densenet161(pretrained=True)
        input_size = model.classifier.in_features
    else:
        raise ValueError('Unexpected network architecture', model)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
     
    # There are 102 flower categories
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size, hidden_units)), \
                                            ('relu1', nn.ReLU()), \
                                            ('drop1', nn.Dropout(p=0.33)), \
                                            ('fc2', nn.Linear(hidden_units, 102)), \
                                            ('output', nn.LogSoftmax(dim=1))]))
    
    criterion = nn.NLLLoss() 
    classifier_models = ["alexnet", "vgg13", "vgg13_bn", "densenet161"]
    if model_name in classifier_models:
        model.classifier = classifier
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    else:
        # Model is resnet
        model.fc = classifier
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    print("Model Created")
    return model, criterion, optimizer, input_size

def validation_evaluation(model, validloader, gpu=False):
    """
    Tracks the current accuracy and loss on the validation dataset.
    """
    correct = 0
    total = 0
    validation_loss = 0
    # Stop autograd from tracking history on Tensors 
    with torch.no_grad():
        for images, labels in validloader:
            # If use gpu and gpu is available
            if gpu and torch.cuda.is_available():
                images, labels = images.to('cuda'), labels.to('cuda')
            else:
                images, labels = images.to('cpu'), labels.to('cpu')
            #outputs = model.forward(images)
            outputs = model(images)
            validation_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    validation_accuracy = correct/total
    validation_loss /= total
    return validation_loss, validation_accuracy

def train_model(model, trainloader, validloader, epochs, print_every, criterion, optimizer, gpu=False):
    """
    Train a new feed-forward classifier using features from a pre-trained network (AlexNet, VGG13, 
    VGG13_BN, ResNet34, Densenet121). Also tracks the accuracy and loss on both the training and
    validation dataset.
    """
    print("Starting Training")
    steps = 0
    # If use gpu and gpu is available
    if gpu and torch.cuda.is_available():
        model.to('cuda')
    else:
        model.to('cpu')
    for e in range(epochs):
        running_loss = 0
        correct = 0
        total = 0              
        for inputs, labels in trainloader:
            steps += 1
            # If use gpu and gpu is available
            if gpu and torch.cuda.is_available():
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            else:
                inputs, labels = inputs.to('cpu'), labels.to('cpu')
            model.train()
            optimizer.zero_grad()
            # Forward and backward passes
            #outputs = model.forward(inputs)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Update training loss
            running_loss += loss.item()
            
            # Update training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()              
                validation_loss, validation_accuracy = \
                validation_evaluation(model, validloader, gpu)
                print("Epoch: {}/{}... ".format(e+1, epochs), \
                      "Training Loss: {:.4f}...".format(running_loss/print_every), \
                      "Validation Loss: {:.4f}...".format(validation_loss), \
                      "Training Acc: {:.4f}...".format(correct/total), \
                      "Validation Acc: {:.4f}...".format(validation_accuracy))
                running_loss = 0    
                # Return model to training mode
                model.train()
    print("Model Trained")

def test_evaluation(model, testloader, gpu=False):
    """
    Returns accuracy on the test dataset
    """
    # If use gpu and gpu is available
    if gpu and torch.cuda.is_available():
        model.to('cuda')
    else:
        model.to('cpu')
    correct = 0
    total = 0
    # Stop autograd from tracking history on Tensors 
    with torch.no_grad():
        for images, labels in testloader:
            # If use gpu and gpu is available
            if gpu and torch.cuda.is_available():
                images, labels = images.to('cuda'), labels.to('cuda')
            else:
                images, labels = images.to('cpu'), labels.to('cpu')
            outputs = model.forward(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = correct/total
    print("Test Accuracy: {:.4f}...".format(test_accuracy))
    return test_accuracy

def save_checkpoint(arch, model, class_to_idx, criterion, optimizer, input_size, hidden_units, epochs, save_dir):
    """ Save a trained model so that it can be used later
    """
    model.class_to_idx = class_to_idx
    checkpoint = {'arch': arch, \
                  'class_to_idx': model.class_to_idx, \
                  'criterion': criterion, \
                  'optimizer': optimizer.state_dict, \
                  'input_size': input_size, \
                  'hidden_units': hidden_units, \
                  'epochs': epochs, \
                  'state_dict': model.state_dict()}
    if len(save_dir) == 0:
        filename = "checkpoint.pth"
    else:
        filename = save_dir + '/checkpoint.pth'
    torch.save(checkpoint, filename)
    print("Model Saved")

if __name__ == '__main__':
    args = process_argument()
    learning_rate, hidden_units, epochs = args.learning_rate, args.hidden_units, args.epochs
    data_directory = args.data_dir
    save_directory = args.save_dir
    gpu = args.gpu
    train_dataloaders, valid_dataloaders, test_dataloaders, class_to_idx = create_dataloaders(data_directory)
    model, criterion, optimizer, input_size = create_model(args.arch, hidden_units, learning_rate)
    train_model(model, train_dataloaders, valid_dataloaders, epochs, 40, criterion, optimizer, gpu=False)
    test_accuracy = test_evaluation(model, test_dataloaders, gpu)
    save_checkpoint(args.arch, model, class_to_idx, criterion, optimizer, input_size, hidden_units, epochs, save_directory)
    
    # cd paind-project
    # python train.py --arch alexnet --learning_rate 0.01 --hidden_units 256 --epochs 1 --gpu ../aipnd-project/flowers
    