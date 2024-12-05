""" benchmarking file """

# Imports
import argparse
import csv
import os
import pickle
import resource

import monai
import torch
from IPython.display import clear_output
from monai.utils import set_determinism

from utils import (convert2binary, create_tensorboard, get_data_and_transforms,
                   get_loaders, get_model_setup, init_weights, test, pretrain, train)

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.cuda.empty_cache()
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

# Set arguments of experiment

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

def parse_args() -> argparse.Namespace:
    """Argument parser

    Returns:
        parser: returns elements in a struct
    """

    parser = argparse.ArgumentParser(description='Process some integers.')

    # Adding arguments
    parser.add_argument('--binary_classes', action = 'store_true',
                        help = 'a boolean flag for binary classes')
    parser.add_argument('--mods_keep', type = validate_mods_keep, default = ["CT1", "T1", "T2", "FLAIR"],
                        help = 'a list of strings, defaults to ["CT1", "T1", "T2", "FLAIR"]')
    parser.add_argument('--model_name', type = str, required = True,
                        help = 'the name of the model, mandatory argument')
    parser.add_argument('--n_folds', type = int, default = 5,
                        help = 'an integer for number of folds, defaults to 5')
    parser.add_argument('--subtract', action = 'store_true',
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
    parser.add_argument('--decrease_LR', action = 'store_true',
                        help = 'a boolean flag for subtract, defaults to True')
    parser.add_argument('--dec_LR_factor', type = int, default = 10,
                        help = 'an integer for dec_LR_factor, defaults to 10')
    parser.add_argument('--loss_weight', type = str, default = "1/prev",
                        help = 'loss function weight on classes, defaults to 1/prevalence weights. Options are: equal, 1-prev, 1/prev.')
    parser.add_argument('--loss', type = str, default = "CE",
                        help = 'loss function, defaults to Cross entropy. Options are: CE, FL.')
    parser.add_argument('--sampler_weight', type = str, default = "1-prev",
                        help = 'loss function weight on classes, defaults to 1-prevalence of class due to the class imbalance. Options are: equal, 1-prev, 1/prev.')
    parser.add_argument('--pretraining', type = int, default = 0,
                        help = 'which pretraining to use')
    parser.add_argument('--clinical_data', action = 'store_true',
                        help = 'whether to use clinical data')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    arguments = parse_args()
    print(arguments.mods_keep)

    MAIN_DIR       = os.getcwd()
    DATASET        = "rano_" + "_".join(arguments.mods_keep) + "_T-1"
    CLASSES        = ["PD", "SD", "PR", "CR"]
    NUM_CLASSES    = len(CLASSES)
    LOGS_FOLDER    = "./Pytorch_Logs"
    DATA_DIR       = os.path.join("./LUMIERE/Imaging", DATASET)
    MODEL_NAME     = arguments.model_name
    BS             = arguments.batch_size
    N_EPOCHS       = arguments.n_epochs
    LEARNING_RATE  = arguments.learning_rate
    WEIGHT_DECAY   = arguments.weight_decay
    PATIENCE       = arguments.patience
    LOSS_WEIGHT    = arguments.loss_weight
    SUBTRACT       = arguments.subtract
    SAMPLER_WEIGHT = arguments.sampler_weight
    CONVERT_BIN    = arguments.binary_classes
    DECREASE_LR    = arguments.decrease_LR
    DEC_LR_FACTOR  = arguments.dec_LR_factor
    LOSS           = arguments.loss
    PRETRAINING    = arguments.pretraining
    CLINICAL_DATA  = arguments.clinical_data

    SEED = 2
    set_determinism(SEED)

    # Get table with data info
    print("Getting data...")
    with open(os.path.join(MAIN_DIR, "table_classifiable.pkl"), "rb") as f:
        TABLE_CLASSIFIABLE = pickle.load(f)

    # Create the data, get the transforms, the number of channels and the labels
    data,  transforms_train, transforms_test, \
        num_channels, labels = get_data_and_transforms(DATA_DIR, TABLE_CLASSIFIABLE, CLASSES, SUBTRACT, CLINICAL_DATA)

    if CONVERT_BIN:
        data, labels, CLASSES, NUM_CLASSES, dataset = convert2binary(data, labels, DATASET)

    # Divide data into train and test set and create each data loader
    print("Creating train and validation datasets...")
    folds = monai.data.utils.partition_dataset_classes(data, labels, num_partitions = arguments.n_folds, seed = SEED)

    # Create folds
    for fold in range(arguments.n_folds):
        print("Creating loaders...")
        class_prevalence, train_loader, test_loader = get_loaders(CLASSES, BS,
                                                                  SAMPLER_WEIGHT, transforms_train,
                                                                  transforms_test, fold, folds)
        print("Creating model...")
        MODEL_CONFIG, model, WEIGHT, OPTIMIZER, LOSS_FUNCTION = get_model_setup(MODEL_NAME,
                                                                                class_prevalence,
                                                                                DEVICE,
                                                                                LEARNING_RATE,
                                                                                WEIGHT_DECAY,
                                                                                LOSS_WEIGHT,
                                                                                num_channels,
                                                                                NUM_CLASSES)
        # Add weights to model
        model.apply(init_weights)
        model = pretrain(model, PRETRAINING, DEVICE, LOGS_FOLDER, MAIN_DIR, num_channels)
        clear_output()

        # Create logs
        LOG_DIR, writer = create_tensorboard(N_EPOCHS, BS, LEARNING_RATE, LOGS_FOLDER,
                                             DEVICE, MODEL_NAME, DATASET, SUBTRACT,
                                             WEIGHT_DECAY, WEIGHT, LOSS_FUNCTION,
                                             DECREASE_LR, SAMPLER_WEIGHT, DEC_LR_FACTOR,fold)

        print("Learning with model " + MODEL_NAME)
        train(LOG_DIR, writer, train_loader, test_loader, MODEL_NAME,
              DATASET, DEVICE, arguments.learning_rate, N_EPOCHS, OPTIMIZER,
              SEED, WEIGHT_DECAY, LOSS_FUNCTION, model, DECREASE_LR,
              DEC_LR_FACTOR, PATIENCE, CLINICAL_DATA)

        # Test model with unseen data
        print("----------")
        print("Testing model with unseen data...")
        result, _, _ = test(MODEL_CONFIG, LOG_DIR, DATASET, DEVICE, BS,
                            CLASSES, test_loader, MODEL_NAME, CLINICAL_DATA)

        # Add to the created dictionary the parameters of the experiment
        result["Binary"]         = CONVERT_BIN
        result["Modalities"]     = DATASET
        result["Fold"]           = fold
        result["Subtract"]       = SUBTRACT
        result["n_epochs"]       = N_EPOCHS
        result["learning_rate"]  = LEARNING_RATE
        result["batch_size"]     = BS
        result["model"]          = MODEL_NAME
        result["loss_weight"]    = LOSS_WEIGHT
        result["sampler_weight"] = SAMPLER_WEIGHT
        result["loss"]           = arguments.loss
        result["pretraining"]    = arguments.pretraining
        result["clinical_data"]  = arguments.clinical_data

        # Save results to csv
        print(result)
        with open('results.csv', 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(result.values())
        print("Ended Fold "+str(fold+1)+"/"+str(arguments.n_folds))
