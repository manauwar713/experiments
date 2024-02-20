import torch
import math
import torch.nn.functional as F
from torch import nn
from torch import optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from model import ANNModel
import os



class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))


def train(model, train_loader, criterion, optimizer):
    model.train()
    train_loss = 0.0
    correct = 0

    for data, target in train_loader:
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        prediction = output.argmax(dim=1, keepdim=True)
        correct += prediction.eq(target.view_as(prediction)).sum().item()


    return train_loss / len(train_loader), correct/len(train_loader.dataset)


def validation(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)

            test_loss += criterion(output, target).item()
            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    test_loss /= len(test_loader)
    correct /= len(test_loader.dataset)

    return (test_loss, correct)

def create_model_optimizer_criterion_dict(path,learning_rate):
    model_dict = dict()
    optimizer_dict= dict()
    criterion_dict = dict()

    for file in os.listdir(path):
        model_name= file
        model_info=ANNModel()
        model_dict.update({model_name : model_info })

        optimizer_name=file
        optimizer_info = torch.optim.SGD(model_info.parameters(), lr=learning_rate)
        optimizer_dict.update({optimizer_name : optimizer_info })

        criterion_name = file
        criterion_info = nn.BCELoss()
        criterion_dict.update({criterion_name : criterion_info})

    return model_dict, optimizer_dict, criterion_dict

def get_averaged_weights(path,model_dict, num_clients):

    fc1_mean_weight = torch.zeros(size=model_dict['clien1.csv'].fc1.weight.shape)
    fc1_mean_bias = torch.zeros(size=model_dict['client1.csv'].fc1.bias.shape)

    fc2_mean_weight = torch.zeros(size=model_dict['client1.csv'].fc2.weight.shape)
    fc2_mean_bias = torch.zeros(size=model_dict['client1.csv'].fc2.bias.shape)

    fc3_mean_weight = torch.zeros(size=model_dict['client1.csv'].fc3.weight.shape)
    fc3_mean_bias = torch.zeros(size=model_dict['client1.csv'].fc3.bias.shape)





    with torch.no_grad():


        for file in os.listdir(path):
            fc1_mean_weight += model_dict[file].fc1.weight.data.clone()
            fc1_mean_bias += model_dict[file].fc1.bias.data.clone()

            fc2_mean_weight += model_dict[file].fc2.weight.data.clone()
            fc2_mean_bias += model_dict[file].fc2.bias.data.clone()

            fc3_mean_weight += model_dict[file].fc3.weight.data.clone()
            fc3_mean_bias += model_dict[file].fc3.bias.data.clone()




        fc1_mean_weight =fc1_mean_weight/num_clients
        fc1_mean_bias = fc1_mean_bias/ num_clients

        fc2_mean_weight =fc2_mean_weight/num_clients
        fc2_mean_bias = fc2_mean_bias/ num_clients

        fc3_mean_weight =fc3_mean_weight/num_clients
        fc3_mean_bias = fc3_mean_bias/ num_clients



    return fc1_mean_weight, fc1_mean_bias, fc2_mean_weight, fc2_mean_bias, fc3_mean_weight, fc3_mean_bias

def set_averaged_weights_as_main_model_weights_and_update_main_model(main_model,model_dict, num_clients):
    fc1_mean_weight, fc1_mean_bias, fc2_mean_weight, fc2_mean_bias, fc3_mean_weight, fc3_mean_bias = get_averaged_weights(model_dict, num_clients=num_clients)
    with torch.no_grad():
        main_model.fc1.weight.data = fc1_mean_weight.data.clone()
        main_model.fc2.weight.data = fc2_mean_weight.data.clone()
        main_model.fc3.weight.data = fc3_mean_weight.data.clone()


        main_model.fc1.bias.data = fc1_mean_bias.data.clone()
        main_model.fc2.bias.data = fc2_mean_bias.data.clone()
        main_model.fc3.bias.data = fc3_mean_bias.data.clone()

    return main_model

def send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, num_clients):
    with torch.no_grad():
        for i in range(num_clients):

            model_dict[file].fc1.weight.data =main_model.fc1.weight.data.clone()
            model_dict[file].fc2.weight.data =main_model.fc2.weight.data.clone()
            model_dict[file].fc3.weight.data =main_model.fc3.weight.data.clone()


            model_dict[file].fc1.bias.data =main_model.fc1.bias.data.clone()
            model_dict[file].fc2.bias.data =main_model.fc2.bias.data.clone()
            model_dict[file].fc3.bias.data =main_model.fc3.bias.data.clone()


    return model_dict

def start_train_end_node_process(num_clients, print_amount):
    for i in range (num_clients):

        train_ds = TensorDataset(x_train_dict[name_of_x_train_sets[i]], y_train_dict[name_of_y_train_sets[i]])
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        test_ds = TensorDataset(x_test_dict[name_of_x_test_sets[i]], y_test_dict[name_of_y_test_sets[i]])
        test_dl = DataLoader(test_ds, batch_size= batch_size * 2)

        model=model_dict[file]
        criterion=criterion_dict[name_of_criterions[i]]
        optimizer=optimizer_dict[name_of_optimizers[i]]

        if i<print_amount:
            print("Subset" ,i)

        for epoch in range(numEpoch):

            train_loss, train_accuracy = train(model, train_dl, criterion, optimizer)
            test_loss, test_accuracy = validation(model, test_dl, criterion)

            if i<print_amount:
                print("epoch: {:3.0f}".format(epoch+1) + " | train accuracy: {:7.5f}".format(train_accuracy) + " | test accuracy: {:7.5f}".format(test_accuracy))

