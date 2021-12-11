import time
import csv
import torch
import torch.nn as nn

class VOCDetector(nn.Module):
    def __init__(self, pretrained_feature_extractor, num_classes):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            *list(pretrained_feature_extractor.features.children())[:-10], # pool4 layer
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(), # Add a latten layer
        )
        for param in self.feature_extractor.parameters(): # Freeze the model
            param.requires_grad = False
            
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
        )
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

# class VGG16FeatureExtractor(nn.Module):
#     def __init__(self, model):
#         super().__init__()
        
#     def forward(self, x):
#         x = self.layers(x)
#         return x

def train_detector(dataloader, model, loss_fn, optimizer, device='cpu'):
    start_time = time.time()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0.0
    model.train()
    for batch, (data, label) in enumerate(dataloader):
        optimizer.zero_grad()
        data, label = data.to(device), label.to(device)
        pred = model(data)
        loss = loss_fn(pred, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch%10 == 0:
            loss, current = loss.item(), batch*len(data)
            print(f'loss: {loss:8f}  [{current:4d}/{size:4d} = {(100*current/size):4.1f}%], batch {batch}, time: {(time.time() - start_time):0.1f} s')
    train_loss /= num_batches
    print(f'Avg loss: {train_loss:8f}')
    return train_loss

def test_detector(dataloader, model, loss_fn, device='cpu'):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    val_loss, correct = 0.0, 0
    with torch.no_grad():
        for data, label in dataloader:
            data, label = data.to(device), label.to(device)
            pred = model(data)
            val_loss += loss_fn(pred, label).item()
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()
    val_loss /= num_batches
    correct /= size
    print(f'Val Error: \n Accuracy: {(100*correct):0.6f}%, Avg loss: {val_loss:0.6f}')
    return correct, val_loss

def write_csv_header(model_index):
    with open('log/' + model_index + '.csv', mode='a') as param_file:
        param_writer = csv.writer(param_file, delimiter=',')
        param_writer.writerow(['Epoch', 'Val accuracy', 'Val loss', 'Train loss', 'Lr'])
        
def write_csv_params(model_index, row):
    with open('log/' + model_index + '.csv', mode='a') as param_file:
        param_writer = csv.writer(param_file, delimiter=',')
        param_writer.writerow(row)