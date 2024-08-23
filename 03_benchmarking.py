#%% 
# Imports
print("Importing...")
import argparse
import csv
import os
import pickle
import resource
import socket
import sys
import time
from datetime import datetime

import monai
import numpy as np
import torch
from IPython.display import clear_output
from monai.data import DataLoader, Dataset
from monai.metrics import ROCAUCMetric, compute_roc_auc
from monai.networks.nets import *
from monai.utils import first, set_determinism
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score)
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from models import *
from utils import critical_error, get_data_and_transforms, probs2logits

cuda = torch.cuda.is_available()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.cuda.empty_cache()
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

#%%
# Set arguments of try
def validate_mods_keep(mods):
    """Function used to validate if the list of mods is usable to get the dataset that we want

    Args:
        mods (str): input string from command

    Raises:
        argparse.ArgumentTypeError: Raises error if the list of mods will not get the dataset correctly

    Returns:
        list: list of modalities
    """
    l = mods.split(",")
    if l not in [["CT1", "T1", "T2", "FLAIR"], ["CT1", "FLAIR"], ["CT1"], ["T1", "FLAIR"], ["T1", "T2", "FLAIR"]]:
        raise argparse.ArgumentTypeError(f"Invalid set of modalities: {l}. Write another set, perhaps in a different order")
    return mods.split(",")

def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')

    # Adding arguments
    parser.add_argument('--binary_classes', action = 'store_true', default = False,
                        help = 'a boolean flag for binary classes')
    parser.add_argument('--mods_keep', type = validate_mods_keep, default = ["CT1", "T1", "T2", "FLAIR"],
                        help = 'a list of strings, defaults to ["CT1", "T1", "T2", "FLAIR"]')
    parser.add_argument('--model_name', type = str, required = True,
                        help = 'the name of the model, mandatory argument')
    parser.add_argument('--n_folds', type = int, default = 5,
                        help = 'an integer for number of folds, defaults to 5')
    parser.add_argument('--subtract', action = 'store_true', default = False,
                        help = 'a boolean flag for subtract, defaults to True')
    parser.add_argument('--n_epochs', type = int, default = 100,
                        help = 'an integer for number of epochs, defaults to 100')
    parser.add_argument('--learning_rate', type = int, default = 1e-4,
                        help = 'an integer for learning rate, defaults to 1e-4')
    parser.add_argument('--batch_size', type = int, default = 4,
                        help = 'an integer for batch size, defaults to 4')
    parser.add_argument('--weight_decay', type = int, default = 0.01,
                        help = 'an integer for weight_decay, defaults to 0.01')
    parser.add_argument('--patience', type = int, default = 10,
                        help = 'an integer for patience, defaults to 10')
    parser.add_argument('--decrease_LR', action = 'store_true', default = False,
                        help = 'a boolean flag for subtract, defaults to True')
    parser.add_argument('--stop_decrease', action = 'store_true', default = False,
                        help = 'a boolean flag for stop_decrease, defaults to True')
    parser.add_argument('--dec_LR_factor', type = int, default = 10,
                        help = 'an integer for dec_LR_factor, defaults to 10')
    parser.add_argument('--loss_weight', type = str, default = "equal",
                        help = 'loss function weight on classes, defaults to equal weights. Options are: equal, 1-prev, 1/prev.')
    parser.add_argument('--sampler_weight', type = str, default = "1-prev",
                        help = 'loss function weight on classes, defaults to 1-prevalence of class due to the class inbalance. Options are: equal, 1-prev, 1/prev.')
    
    args = parser.parse_args()
    return args



#%% Create dataset to classify image types 

# Convert to binary labels
def convert2binary(data, labels):
    for i in range(len(labels)):
        if labels[i] != 0:
            labels[i] = 1
            data[i]["label"] = torch.tensor([0, 1], dtype = torch.float32)
        else:
            data[i]["label"] = torch.tensor([1, 0], dtype = torch.float32)
    classes      = ["PD", "other"]
    num_classes  = len(classes)
    dataset      = dataset + "_bin"
    return classes, num_classes, dataset

def get_loaders(classes, bs, sampler_weight, transforms_train, transforms_test, i, folds):
    train_data_unflattened = folds[0:i]+folds[i+1:]
    train_data = [x for xs in train_data_unflattened for x in xs]
    test_data = folds[i]
    
    print(len(train_data))
    print(len(test_data))

    # Define class prevalence
    element_counts = [0.]*len(classes)
    for d in train_data:
        label = d["label"].argmax()
        element_counts[label] += 1

    class_prevalence = [i/sum(element_counts) for i in element_counts]
    print("Class prevalence: " + str(class_prevalence))

    # Create datasets
    train_ds = Dataset(data = train_data, transform = transforms_train)
    test_ds  = Dataset(data = test_data, transform = transforms_test)

    # Calculate weights for each sample in train_data. Can be 1-prevalence or 1/prevalence
    if sampler_weight!="equal":
        weights = []
        for d in train_data:
            label = d["label"].argmax()
            if sampler_weight == "1/prev":
                weight = 1.0 / class_prevalence[label]
            elif sampler_weight == "1-prev":
                weight = 1.0 - class_prevalence[label] 
            weights.append(weight)

        # Create WeightedRandomSampler
        weights = torch.DoubleTensor(weights)
        sampler = WeightedRandomSampler(weights, len(weights))

        # Create data loaders
        train_loader = DataLoader(train_ds, batch_size = bs, sampler = sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size = bs, shuffle = True)

    test_loader  = DataLoader(test_ds, batch_size = bs, shuffle = True)

    check_data   = first(train_loader)
    print(check_data["images"].shape, check_data["label"])
    return class_prevalence, train_loader, test_loader



# %%
#################
# Model Options #
#################
def get_model_setup(model_name, class_prevalence, device, learning_rate, weight_decay, loss_weight, num_channels, num_classes):
    if model_name == "monai_classifier":
        model_config  = Classifier(in_shape = (num_channels, 240, 240, 155), classes = num_classes, channels = (2, 4, 8), strides = (2, 4, 8), last_act = "softmax")
    elif model_name == "monai_densenet121":
        model_config  = DenseNet121(spatial_dims = 3, in_channels = num_channels, out_channels = num_classes, pretrained = False)
    elif model_name == "monai_densenet169":
        model_config  = DenseNet169(spatial_dims = 3, in_channels = num_channels, out_channels = num_classes, pretrained = False)
    elif model_name == "monai_densenet201":
        model_config  = DenseNet201(spatial_dims = 3, in_channels = num_channels, out_channels = num_classes, pretrained = False)
    elif model_name == "monai_densenet264":
        model_config  = DenseNet264(spatial_dims = 3, in_channels = num_channels, out_channels = num_classes, pretrained = False)
    elif model_name == "monai_vit":
        model_config  = ViT(in_channels = num_channels, img_size = [240, 240, 155], patch_size = [20, 20, 10], classification = True, num_classes = num_classes, pos_embed_type = 'sincos', dropout_rate = 0.1)
    elif model_name == "monai_resnet":
        model_config  = ResNet("bottleneck", (3, 4, 6, 3), (64, 128, 256, 512), spatial_dims = 3, n_input_channels = num_channels, num_classes = num_classes)
    elif model_name == "AlexNet3D":
        model_config  = AlexNet3D(num_channels, num_classes = num_classes)  
    elif model_name == "GoogleNet3D":
        model_config  = GoogleNet3D(num_channels, num_classes = num_classes)  
    else: sys.exit('Please choose one of the models available. You didnt write any one of them')

    model     = model_config.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), learning_rate, weight_decay = weight_decay) # torch.optim.SGD(model.parameters(), learning_rate, weight_decay = weight_decay)

    # Define weight of loss function
    if loss_weight == "equal": 
        weight = None
    elif loss_weight == "1/prev":
        weight = torch.tensor([1/a for a in class_prevalence], dtype = torch.float32).to(device)
    elif loss_weight=="1-prev":
        weight = torch.tensor([1-a for a in class_prevalence], dtype = torch.float32).to(device)

    # Define loss function
    if num_classes ==2:
        loss_function = torch.nn.BCEWithLogitsLoss(weight)
    else: 
        loss_function = torch.nn.CrossEntropyLoss(weight = weight)
    
    return model_config, model, weight, optimizer, loss_function


def init_weights(model):
    """Initializes the weights of an architecture

    Args:
        model (obj): model to be initiated
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Conv3d) or isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 1)
        elif isinstance(module, torch.nn.BatchNorm3d):
            torch.nn.init.constant_(module.weight, 1)
            torch.nn.init.constant_(module.bias, 0)





#%% 
# start a typical PyTorch training



def create_tensorboard(n_epochs, bs, learning_rate, logs_folder, 
                       device, model_name, dataset, subtract, weight_decay, 
                       weight, loss_function, stop_decrease, decrease_LR, sampler_weight, dec_LR_factor,fold):
    
    """This function generates a tensorboard folder to keep the evolution of the model training

    Args:
        n_epochs (int): number of (maximum) epochs
        bs (int): batch size
        learning_rate (float): learning rate for the optimizer
        logs_folder (str): folder where the logs will be saved
        device (str): name of device
        model_name (str): name of model to use
        dataset (str): name of to get dataset
        subtract (bool): whether to subtract images of consecutive timepoints
        weight_decay (float): weight decay for adamW
        weight (tensor): weight of loss function (one value per class)
        loss_function (obj): Loss function to use
        stop_decrease (bool): Whether to stop the learning process
        decrease_LR (bool): Whether to decrease the learning rate
        sampler_weight (str): How to perform the sampler weight (1-prevalence or 1/prevalence)
        dec_LR_factor (int): Factor by which to decrease the learning rate

    Returns:
        list: log dir of particular training session, and the writer
    """
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = os.path.join(logs_folder, current_time + "_" + host)
    print("Log: " + log_dir)
    writer = SummaryWriter(log_dir = log_dir)
    writer.add_text("Number of epochs", "Number of epochs = " + str(n_epochs))
    writer.add_text("Batch Size", "Batch Size = " + str(bs))
    writer.add_text("Learning Rate", "Learning Rate = " + str(learning_rate))
    writer.add_text("Device", "Device: " + str(device))
    writer.add_text("Model", "Architecture: " + model_name)
    writer.add_text("Dataset", "Dataset: " + dataset)
    writer.add_text("Subtract", "Subtract: " + str(subtract))
    writer.add_text("weight_decay", "weight_decay: " + str(weight_decay))
    writer.add_text("loss function weight", "weight: " + str(weight))
    writer.add_text("loss_function", "loss_function: " + str(loss_function))
    writer.add_text("stop_decrease", "stop_decrease: " + str(stop_decrease))
    writer.add_text("decrease_LR", "decrease_LR: " + str(decrease_LR))
    writer.add_text("sampler_weight", "sampler_weight: " + sampler_weight)
    writer.add_text("dec_LR_factor", "decrease LR factor: " + str(dec_LR_factor))
    writer.add_text("fold", "fold: " + str(fold))
    return log_dir, writer



def train(log_dir, writer, train_loader, test_loader, model_name, 
          device, learning_rate, n_epochs, optimizer, seed, weight_decay, 
          loss_function, model, stop_decrease, decrease_LR, patience):
    """Training and validation loop

    Args:
        log_dir (str): log directory
        writer (obj): writer object of tensorflow
        train_loader (dataloader): dataloader of training data
        test_loader (dataloader): dataloader of test data
        model_name (str): name of the model to use
        device (str): name of device
        learning_rate (float): learning rate for the optimizer
        n_epochs (int): number of (maximum) epochs
        optimizer (obj): optimizer
        seed (int): seed number
        weight_decay (float): weight decay for adamW
        val_interval (int): interval of epochs to perform validation
        loss_function (obj): Loss function to use
        model (obj): model object
        stop_decrease (bool): Whether to stop the learning process
        decrease_LR (bool): Whether to decrease the learning rate
        patience (int): _description_
    """
    torch.cuda.empty_cache()
    best_metric         = -1
    best_metric_epoch   = -1
    auc_metric          = ROCAUCMetric()
    epoch_len           = len(train_loader)
    best_val_loss       = np.inf
    best_train_loss     = np.inf
    lr_decreases        = 0
    best_acc            = 0
    epoch_loss          = np.inf

    torch.save(model.state_dict(), os.path.join(log_dir, "checkpoint" + ".pt"))
    os.chmod(log_dir,0o777)
    print("Starting training...")
    # Run learning
    torch.cuda.empty_cache()
    for epoch in range(n_epochs):
        # Early stopping if lr decreases more then patience times
        if stop_decrease and lr_decreases == patience:
           break

        print("-" * 10)
        print(f"epoch {epoch + 1}/{n_epochs}")

        model.train()
        epoch_loss = 0
        step = 0
        start_epoch = time.time()
        last_batch = time.time()
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["images"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels) if arguments.model_name != "monai_vit" else loss_function(outputs[0], labels)
        
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
            end_batch = time.time()
            time_batch = (end_batch-last_batch)
            last_batch = end_batch
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}, time: {time_batch:.2f}s")

        epoch_loss /= step # mean loss
        end_epoch = time.time()
        time_epoch = np.round((end_epoch-start_epoch)/60, 2)
        print(f"epoch {epoch + 1} average training loss: {epoch_loss:.4f}, epoch duration: {time_epoch:.2f}min")
    
        writer.add_scalar("train_loss", epoch_loss, epoch)
    
        # Create new checkpoint or go back to previous checkpoint and diminish learning rate/change seed if necessary
        if epoch_loss<best_train_loss: # if loss improves, save checkpoint and go back to 0 patience
            torch.save(model.state_dict(), os.path.join(log_dir, "checkpoint" + ".pt"))
            best_train_loss = epoch_loss
            lr_decreases = 0
        # if loss stays the same or increases, load last checkpoint and either decrease LR or create new seed so that training does not put us on the same spot again
        else:
            model.load_state_dict(torch.load(os.path.join(log_dir, "checkpoint" + ".pt"), map_location = device))
            if decrease_LR:
                learning_rate /= dec_LR_factor
                optimizer = torch.optim.AdamW(model.parameters(), learning_rate, weight_decay = weight_decay)
            else:
                seed += 1
                set_determinism(seed)

            # increase patience
            lr_decreases += 1
            print("Diminished learning rate (" + str(lr_decreases) + "/"+str(patience)+")")
        
        # Validation
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            y_pred = torch.tensor([], dtype = torch.float32, device = device)
            y_pred_onehot = torch.tensor([], dtype = torch.float32, device = device)
        
            y = torch.tensor([], dtype = torch.long, device = device)
            y_onehot = torch.tensor([], dtype = torch.long, device = device)
            for i, val_data in enumerate(test_loader):
                # Get ground truth and calculate output given input
                val_images, val_labels = val_data["images"].to(device), val_data["label"].to(device)
                outputs = model(val_images)
                output_onehot = probs2logits(outputs, device = device) if model_name != "monai_vit" else probs2logits(outputs[0],device = device)
                y_onehot = torch.cat([y_onehot, val_labels], dim = 0)
                y = torch.cat([y, torch.argmax(val_labels, 1)], dim = 0)
            
                # Calculate loss
                loss = loss_function(outputs, val_labels) if model_name != "monai_vit" else loss_function(outputs[0], val_labels)
                val_loss += loss
            
                # Calculate predicted outputs
                y_pred = torch.cat([y_pred, torch.argmax(output_onehot, 1)], dim = 0)
                y_pred_onehot = torch.cat([y_pred_onehot, output_onehot], dim = 0)

            # Calculate metrics
            val_loss /= (i + 1)
            acc_value = balanced_accuracy_score(y.cpu().numpy(), y_pred.cpu().numpy())
            auc_metric(y_pred_onehot, y_onehot)
            auc_result = auc_metric.aggregate()
            auc_metric.reset()

            print(f"epoch {epoch + 1} average validation loss: {val_loss:.4f}")

            # Save if better validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metric = best_val_loss
                best_metric_epoch = epoch + 1
                best_acc = acc_value

                if os.path.exists(os.path.join(log_dir, dataset + ".pt")): os.remove(os.path.join(log_dir, dataset + ".pt"))
                torch.save(model.state_dict(), os.path.join(log_dir, dataset + ".pt"))
                print("Saved new best metric model according to loss")

            print("current epoch: {}; current accuracy: {:.4f}; current AUC: {:.4f}; best loss: {:.4f}, with acc= {:.4f} at epoch {}".format(epoch + 1, acc_value, auc_result, best_metric, best_acc, best_metric_epoch))
        
            # Write to writer
            writer.add_scalar("val_accuracy", acc_value, epoch + 1)
            writer.add_scalar("AUC", auc_result, epoch + 1)
            writer.add_scalar("val_loss", val_loss, epoch + 1)

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    os.remove(os.path.join(log_dir, "checkpoint" + ".pt"))
    writer.close()


# %%


# Load the model and put it in evaluation mode
def test(model_config, log_dir, dataset, device, bs, classes, test_loader, model_name):
    model_trained = model_config
    print("Loading weights from: "+os.path.join(log_dir, dataset + ".pt"))
    model_trained.load_state_dict(torch.load(os.path.join(log_dir, dataset + ".pt"), map_location = device))

    model_trained.eval()
    model_trained.to(device)

    if len(classes) == 2:
        sigsoft = nn.Sigmoid()
    else:
        sigsoft = nn.Softmax(dim = 1)

    # Loop through the testing images
    with torch.no_grad():
        predicted = torch.tensor([], dtype = torch.float32, device = device)
        predicted_probs = torch.empty((bs, len(classes)), dtype = torch.float32, device = device)
        real_onehot = torch.empty((bs, len(classes)), dtype = torch.long, device = device)
        real = torch.tensor([], dtype = torch.long, device = device)
        for batch in test_loader:
            # Get ground truth and calculate output given input
            test_images, test_labels = batch["images"].to(device), batch["label"].to(device)
            real = torch.cat([real, test_labels.argmax(dim = 1)])
            real_onehot = torch.cat([real_onehot, test_labels], dim = 0)
        
            # Get output/predicted values
            outputs_probs = sigsoft(model_trained(test_images)) if model_name != "monai_vit" else sigsoft(model_trained(test_images)[0])
            outputs = outputs_probs.argmax(dim = 1)
            predicted = torch.cat([predicted, outputs])
            predicted_probs = torch.cat([predicted_probs, outputs_probs], dim = 0)

    predicted_probs = predicted_probs[bs:]  
    real_onehot = real_onehot[bs:]  

    real = real.cpu().numpy()
    predicted = predicted.cpu().numpy()

    # Calculate performance metrics
    precision = precision_score(real, predicted, zero_division = 0, average = 'weighted')
    recall = recall_score(real, predicted, zero_division = 0, average = 'weighted')
    bal_accuracy = balanced_accuracy_score(real, predicted)
    accuracy = accuracy_score(real, predicted)
    F1_score = 2 * (precision * recall) / (precision + recall)
    critical_err = critical_error(predicted, real)

    print("Balenced Accuracy    :", bal_accuracy)
    print("Accuracy             :", accuracy)
    print("F1-score             :", F1_score)
    print("Precision            :", precision)
    print("Recall               :", recall)
    print("Critical Error Rate  :", critical_err)

    # Calculate ROC AUC
    roc_auc = compute_roc_auc(predicted_probs, real_onehot)
    print(f"ROC AUC: {roc_auc}")

    # Create results variable and save it in a pickle format
    result = {
    "logdir"                : os.path.basename(log_dir),
    "Balenced Accuracy"     : str(bal_accuracy),
    "Accuracy"              : str(accuracy),
    "F1-score"              : str(F1_score),
    "Precision"             : str(precision),
    "Recall"                : str(recall),
    "ROC AUC"               : str(roc_auc),
    "Critical Error Rate"   : str(critical_err)}
    return result, real, predicted

# %%

if __name__ == '__main__':
    arguments = parse_args()
    print(arguments.mods_keep)

    host           = socket.gethostname() if socket.gethostname() == "pangeia" else "pluto"
    phd_dir        = os.getcwd()
    dataset        = "rano_"+"_".join(arguments.mods_keep)+"_T-1"
    print(dataset)
    classes        = ["PD", "SD", "PR", "CR"]
    num_classes    = len(classes)
    logs_folder    = "/laseebhome/amatoso/phd/Pytorch_Logs" if host == "pluto" else "/home/amatoso/phd/Pytorch_Logs"
    data_dir       = os.path.join(phd_dir, "lumiere/datasets/", dataset) if host == "pangeia" else os.path.join(phd_dir, "lumiere/datasets_pluto/", dataset)

    model_name     = arguments.model_name # "monai_densenet169" # monai_vit monai_resnet monai_classifier monai_densenet121 monai_densenet169 monai_densenet264 monai_densenet201 jang_model_mri AlexNet3D GoogleNet3D
    bs             = arguments.batch_size
    n_epochs       = arguments.n_epochs
    learning_rate  = arguments.learning_rate
    weight_decay   = arguments.weight_decay
    patience       = arguments.patience
    loss_weight    = arguments.loss_weight      
    subtract       = arguments.subtract         
    sampler_weight = arguments.sampler_weight   
    convert_bin    = arguments.binary_classes   
    decrease_LR    = arguments.decrease_LR      
    stop_decrease  = arguments.stop_decrease
    dec_LR_factor  = arguments.dec_LR_factor

    seed = 2
    set_determinism(seed)

    # Get table with data info
    print("Getting data...")
    with open(os.path.join(phd_dir, "lumiere/table_all.pkl"), "rb") as f:
        all = pickle.load(f)

    # Create the data, get transforms and number of channels
    data,  transforms_train, transforms_test, num_channels, labels = get_data_and_transforms(data_dir, all, classes, subtract)

    if convert_bin:
        classes, num_classes, dataset = convert2binary(data, labels) 

    # Divide data into train and test set and create each dataloader
    print("Creating train and validation datasets...")
    folds = monai.data.utils.partition_dataset_classes(data, labels, num_partitions = arguments.n_folds, seed = seed)

    for fold in range(arguments.n_folds):
        print("Creating loaders...")
        class_prevalence, train_loader, test_loader = get_loaders(classes, bs, sampler_weight, transforms_train, transforms_test, fold, folds)

        device = torch.device("cuda" if cuda else "cpu")
        print("Creating model...")
        model_config, model, weight, optimizer, loss_function = get_model_setup(model_name, class_prevalence, device, learning_rate, weight_decay, loss_weight, num_channels, num_classes)
        model.apply(init_weights)
        clear_output()

        print("Learning with model "+ model_name)
        log_dir, writer = create_tensorboard(n_epochs, bs, learning_rate, logs_folder, device, model_name, dataset, subtract, weight_decay, weight, loss_function, stop_decrease, decrease_LR, sampler_weight, dec_LR_factor,fold)
        

        train(log_dir, writer, train_loader, test_loader, model_name, device, arguments.learning_rate, n_epochs, optimizer, seed, weight_decay, loss_function, model, stop_decrease, decrease_LR, patience)

        # Test model with unseen data
        print("----------")
        print("Testing model with unseen data...")
        result, _, _ = test(model_config, log_dir, dataset, device, bs, classes, test_loader, model_name)
        result["Binary"] = arguments.binary_classes
        result["Modalaties"] = dataset
        result["Fold"] = fold
        result["Subtract"] = arguments.subtract
        result["n_epochs"] = arguments.n_epochs
        result["learning_rate"] = arguments.learning_rate
        result["batch_size"] = arguments.batch_size
        result["model"] = arguments.model_name
        result["loss_weight"] = arguments.loss_weight
        result["sampler_weight"] = arguments.sampler_weight

        print(result)
        with open('/home/amatoso/phd/results.csv', 'a') as file: 
            writer = csv.DictWriter(file, fieldnames = result.keys())
            writer.writerow(result)
            file.close()
        print("Ended Fold "+str(fold+1)+"/"+str(arguments.n_folds))

