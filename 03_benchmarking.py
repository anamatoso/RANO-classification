""" benchmarking file """

# Imports

import argparse
import csv
import os
import pickle
import resource
import socket

import monai
import torch
from IPython.display import clear_output
from monai.utils import set_determinism

from utils import (convert2binary, create_tensorboard, get_data_and_transforms,
                   get_loaders, get_model_setup, init_weights, test, train)

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
        argparse.ArgumentTypeError: Raises error if the list of 
        mods will not get the dataset correctly

    Returns:
        list: list of modalities
    """
    l = mods.split(",")
    if l not in [["CT1", "T1", "T2", "FLAIR"], ["CT1", "FLAIR"],
                 ["CT1"], ["T1", "FLAIR"], ["T1", "T2", "FLAIR"]]:
        raise argparse.ArgumentTypeError(f"Invalid set of modalities: {l}. "+
                                         "Write another set, perhaps in a different order")
    return mods.split(",")

def parse_args():
    """Argument parser

    Returns:
        _type_: _description_
    """
    parser = argparse.ArgumentParser(description='Process some integers.')

    # Adding arguments
    parser.add_argument('--binary_classes', action = 'store_true', default = False,
                        help = 'a boolean flag for binary classes')
    parser.add_argument('--mods_keep', type = validate_mods_keep, default = ["CT1",
                                                                              "T1", "T2", "FLAIR"],
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
                        help = "loss function weight on classes, defaults to equal weights. "+
                        "Options are: equal, 1-prev, 1/prev.")
    parser.add_argument('--sampler_weight', type = str, default = "1-prev",
                        help = "loss function weight on classes, defaults to 1-prevalence of"+
                        "class due to the class inbalance. Options are: equal, 1-prev, 1/prev.")

    args = parser.parse_args()
    return args
# %%

if __name__ == '__main__':
    arguments = parse_args()
    print(arguments.mods_keep)

    HOST           = socket.gethostname() if socket.gethostname() == "pangeia" else "pluto"
    PHD_DIR        = os.getcwd()
    DATASET        = "rano_"+"_".join(arguments.mods_keep)+"_T-1"
    print(DATASET)
    CLASSES        = ["PD", "SD", "PR", "CR"]
    NUM_CLASSES    = len(CLASSES)
    LOGS_FOLDER    = "/laseebhome/amatoso/phd/Pytorch_Logs" if HOST == "pluto" \
        else "/home/amatoso/phd/Pytorch_Logs"
    DATA_DIR       = os.path.join(PHD_DIR, "lumiere/datasets/", DATASET) if HOST == "pangeia" \
        else os.path.join(PHD_DIR, "lumiere/datasets_pluto/", DATASET)

    MODEL_NAME     = arguments.model_name
    BS             = arguments.batch_size
    N_EPOCHS       = arguments.n_epochs
    LEARNINIG_RATE = arguments.learning_rate
    WEIGHT_DECAY   = arguments.weight_decay
    PATIENCE       = arguments.patience
    LOSS_WEIGHT    = arguments.loss_weight
    SUBTRACT       = arguments.subtract
    SAMPLER_WEIGHT = arguments.sampler_weight
    CONVERT_BIN    = arguments.binary_classes
    DECREASE_LR    = arguments.decrease_LR
    STOP_DECREASE  = arguments.stop_decrease
    DEC_LR_FACTOR  = arguments.dec_LR_factor

    SEED = 2
    set_determinism(SEED)

    # Get table with data info
    print("Getting data...")
    with open(os.path.join(PHD_DIR, "lumiere/table_all.pkl"), "rb") as f:
        TABLE_ALL = pickle.load(f)

    # Create the data, get transforms and number of channels
    data,  transforms_train, transforms_test, \
        num_channels, labels = get_data_and_transforms(DATA_DIR, TABLE_ALL, CLASSES, SUBTRACT)

    if CONVERT_BIN:
        data, labels, CLASSES, NUM_CLASSES, dataset = convert2binary(data, labels, CLASSES, DATASET)

    # Divide data into train and test set and create each dataloader
    print("Creating train and validation datasets...")
    folds = monai.data.utils.partition_dataset_classes(data, labels,
                                                       num_partitions = arguments.n_folds,
                                                       seed = SEED)

    for fold in range(arguments.n_folds):
        print("Creating loaders...")
        class_prevalence, train_loader, test_loader = get_loaders(CLASSES, BS,
                                                                  SAMPLER_WEIGHT, transforms_train,
                                                                  transforms_test, fold, folds)

        DEVICE = torch.device("cuda" if cuda else "cpu")
        print("Creating model...")
        MODEL_CONFIG, model, WEIGHT, OPTIMIZER, LOSS_FUNCTION = get_model_setup(MODEL_NAME,
                                                                                class_prevalence,
                                                                                DEVICE,
                                                                                LEARNINIG_RATE,
                                                                                WEIGHT_DECAY,
                                                                                LOSS_WEIGHT,
                                                                                num_channels,
                                                                                NUM_CLASSES)
        model.apply(init_weights)
        clear_output()

        print("Learning with model "+ MODEL_NAME)
        LOG_DIR, writer = create_tensorboard(N_EPOCHS, BS, LEARNINIG_RATE, LOGS_FOLDER,
                                             DEVICE, MODEL_NAME, DATASET, SUBTRACT,
                                             WEIGHT_DECAY, HOST, WEIGHT, LOSS_FUNCTION,
                                             STOP_DECREASE, DECREASE_LR, SAMPLER_WEIGHT,
                                             DEC_LR_FACTOR,fold)


        train(LOG_DIR, writer, train_loader, test_loader, MODEL_NAME,
              dataset, DEVICE, arguments.learning_rate, N_EPOCHS, OPTIMIZER,
              SEED, WEIGHT_DECAY, LOSS_FUNCTION, model, STOP_DECREASE, DECREASE_LR,
              DEC_LR_FACTOR, PATIENCE)

        # Test model with unseen data
        print("----------")
        print("Testing model with unseen data...")
        result, _, _ = test(MODEL_CONFIG, LOG_DIR, DATASET, DEVICE, BS,
                            CLASSES, test_loader, MODEL_NAME)
        result["Binary"] = CONVERT_BIN
        result["Modalaties"] = DATASET
        result["Fold"] = fold
        result["Subtract"] = SUBTRACT
        result["n_epochs"] = N_EPOCHS
        result["learning_rate"] = LEARNINIG_RATE
        result["batch_size"] = BS
        result["model"] = MODEL_NAME
        result["loss_weight"] = LOSS_WEIGHT
        result["sampler_weight"] = SAMPLER_WEIGHT

        print(result)
        with open('/home/amatoso/phd/results.csv', 'a', encoding=None) as file:
            writer = csv.DictWriter(file, fieldnames = result.keys())
            writer.writerow(result)
            file.close()
        print("Ended Fold "+str(fold+1)+"/"+str(arguments.n_folds))
