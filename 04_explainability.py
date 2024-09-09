print("Importing...")
import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import monai
import numpy as np
import pandas as pd
from captum.attr import DeepLift, Saliency
from captum.attr import visualization as viz
from monai.utils import set_determinism
from sklearn.metrics import confusion_matrix

from models import *
from utils import (convert2binary, get_data_and_transforms, get_loaders,
                   get_model_setup, test)

# CAPTUM
# import torchvision
# from captum.attr import IntegratedGradients
# from captum.attr import NoiseTunnel

CUDA = torch.cuda.is_available()
DATASETS = "./Datasets"
def parse_argument():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--log_folder', type = str, required = True,
                        help = 'the name of the log folder to test (basename)')
    parser.add_argument('--n_folds', type = int, default = 5,
                        help = 'an integer for number of folds, defaults to 5')
    parser.add_argument('--weight_decay', type = int, default = 0.01,
                        help = 'an integer for weight_decay, defaults to 0.01')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    arguments = parse_argument()
    print("Testing on "+arguments.log_folder)
    logs_folder    = "Pytorch_Logs"
    if os.path.exists(os.path.join(logs_folder,arguments.log_folder,"confusion_matrix.csv")):
        print(arguments.log_folder + " already done")
        sys.exit(1)
    
    # Get results from that attempt
    df = pd.read_csv('results.csv')
    row = df[df['logdir'] == arguments.log_folder]
    results = row.to_dict(orient='records')[0]

    # Set main variables
    phd_dir         = os.getcwd()
    dataset         = results["dataset"]
    classes         = ["PD", "SD", "PR", "CR"]
    num_classes     = len(classes)
    model_name      = results["model"]
    subtract        = results["subtract"] == True
    convert_bin     = results["binary"] == "True"
    bs              = int(results["bs"])
    learning_rate   = float(results["LR"])
    sampler_weight  = results["sampler_weight"]
    loss_weight     = results["loss_weight"]
    weight_decay    = 0.01
    seed            = 2
    fold            = int(results["fold"])
    set_determinism(seed)

    # Get table with data info
    print("Getting data...")
    with open(os.path.join(phd_dir, "lumiere/table_all.pkl"), "rb") as f:
        all = pickle.load(f)

    # Create the data, get transforms and number of channels
    data,  transforms_train, transforms_test, num_channels, labels = get_data_and_transforms(DATASETS, all, classes, subtract)

    if convert_bin:
        classes, num_classes, dataset = convert2binary(data, labels) 

    # Divide data into train and test set and create each dataloader
    print("Creating train and validation datasets...")
    folds = monai.data.utils.partition_dataset_classes(data, labels, num_partitions = arguments.n_folds, seed = seed)

    print("Getting loaders...")
    class_prevalence, _, test_loader = get_loaders(classes, bs, sampler_weight, transforms_train, transforms_test, fold, folds)

    device = torch.device("cuda" if CUDA else "cpu")
    print("Getting model setup...")
    model_config, model, _, _, _ = get_model_setup(model_name, class_prevalence, device, learning_rate, weight_decay, loss_weight, num_channels, num_classes)        

    # Test model with unseen data
    print("----------")
    print("Testing model with unseen data...")
    log_dir=os.path.join(logs_folder, arguments.log_folder)

    if not os.path.exists(os.path.join(logs_folder,arguments.log_folder,"confusion_matrix.csv")):
        result, real, predicted = test(model_config, log_dir, dataset, device, bs, classes, test_loader, model_name)
        
        # Calculate confusion matrix
        labels=list(range(len(classes)))
        confusion_mat = confusion_matrix(real,predicted,labels=labels)

        np.savetxt(os.path.join(logs_folder,arguments.log_folder,"confusion_matrix.csv"), confusion_mat, delimiter=",")
    
    # Check if confusion matrix has been calculated else save it
    if os.path.exists(os.path.join(logs_folder,arguments.log_folder,"confusion_matrix.png")):
        print("Confusion matrix already done")
        sys.exit(1)
    else:
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(confusion_mat,cmap="jet")
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        for i in range(confusion_mat.shape[0]):
            for j in range(confusion_mat.shape[1]):
                plt.text(j, i, format(confusion_mat[i, j]),
                        ha="center", va="center",size=20,
                        color="white" if confusion_mat[i, j] > confusion_mat.max() / 2 else "black")

        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.colorbar(label="#samples")
        plt.show()
        plt.savefig(os.path.join(logs_folder,arguments.log_folder,"confusion_matrix.png"))



    dataiter = next(iter(test_loader))
    images, labels = dataiter["images"], dataiter["label"]
    
    # print images
    # plot_image(torchvision.utils.make_grid(images))
    # print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    outputs = model(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                for j in range(4)))
    
    ind = 3
    

    input_image = images[ind].unsqueeze(0)
    labels_index = labels[ind]
    input_image.requires_grad = True
    model.eval()

    def attribute_image_features(algorithm, image, labels_index, **kwargs):
        model.zero_grad()
        tensor_attributions = algorithm.attribute(torch.tensor(image),target=torch.tensor(torch.argmax(labels_index)),**kwargs)
        return tensor_attributions
    
    saliency = Saliency(model)
    grads = saliency.attribute(input_image, target=torch.argmax(labels_index))
    

    # ig = IntegratedGradients(model) # Verified until here
    # attr_ig, delta = attribute_image_features(ig, input_image, labels_index, baselines=input_image * 0, return_convergence_delta=True)
    # attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
    # print('Approximation delta: ', abs(delta))

    # ig = IntegratedGradients(model)
    # nt = NoiseTunnel(ig)
    # attr_ig_nt = attribute_image_features(nt, input_image, baselines=input_image * 0, nt_type='smoothgrad_sq',
    #                                     nt_samples=100, stdevs=0.2)
    # attr_ig_nt = np.transpose(attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    
    dl = DeepLift(model)
    attr_dl = attribute_image_features(dl, input_image, labels_index, baselines=input_image * 0)

    # unnormalized_image=input_image.squeeze()[:,:,:,slice_idx].squeeze()[0,:,:].unsqueeze(0).cpu().detach().numpy()
    # maximum=np.max(np.max(unnormalized_image[0]))
    # minimum=np.min(np.min(unnormalized_image[0]))
    # unnormalized_image[0]=(unnormalized_image[0] - minimum) / (maximum - minimum)

    min_vals = input_image.amin(dim=(2, 3, 4), keepdim=True)
    max_vals = input_image.amax(dim=(2, 3, 4), keepdim=True)
    normalized_image = (input_image - min_vals) / (max_vals - min_vals)
    
    slice_idx=100
    image_number=1
    original_image=np.transpose(normalized_image[0,image_number,:,:,slice_idx].unsqueeze(0).cpu().detach().numpy(), (1, 2, 0))
    grads_plot = np.transpose(grads.squeeze()[:,:,:,slice_idx].squeeze()[0,:,:].unsqueeze(0).cpu().detach().numpy(), (1, 2, 0))
    attr_dl_plot = np.transpose(attr_dl.squeeze()[:,:,:,slice_idx].squeeze()[0,:,:].unsqueeze(0).cpu().detach().numpy(), (1, 2, 0))
    _ = viz.visualize_image_attr(None, original_image, method="original_image", title="Original Image", show_colorbar=True)

    _ = viz.visualize_image_attr(grads_plot, original_image, method="blended_heat_map", sign="absolute_value",show_colorbar=True, title="Overlayed Gradient Magnitudes") # Plots until here work

    # _ = viz.visualize_image_attr(attr_ig, original_image, method="blended_heat_map",sign="all",show_colorbar=True, title="Overlayed Integrated Gradients")

    # _ = viz.visualize_image_attr(attr_ig_nt, original_image, method="blended_heat_map", sign="absolute_value", outlier_perc=10, show_colorbar=True, title="Overlayed Integrated Gradients \n with SmoothGrad Squared")

    _ = viz.visualize_image_attr(attr_dl_plot, original_image, method="blended_heat_map",sign="all",show_colorbar=True, title="Overlayed DeepLift")