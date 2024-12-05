""" Auxiliary file with utils functions
"""

import os
import shutil
import subprocess as sub
import sys
import time
from datetime import datetime

import ants
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from IPython.display import display
from ipywidgets import HBox, IntSlider, Layout, interactive
from monai.data import DataLoader, Dataset
from monai.data.meta_tensor import MetaTensor
from monai.metrics import ROCAUCMetric, compute_roc_auc
from monai.networks.nets import (DenseNet121, DenseNet169,
                                 DenseNet264, ViT)
from monai.transforms import (Compose, ConcatItemsd, DeleteItemsd, LoadImaged,
                              NormalizeIntensityd, RandAdjustContrastd,
                              RandFlipd, RandGaussianNoised,
                              RandScaleIntensityd, SubtractItemsd)
from monai.utils import first, set_determinism
from nipype.interfaces.image import Reorient
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score)
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from models import *


def plot_slices(images):

    """Plots a 3x3 grid with the overlay of images in images with 3 slices, 
    one for each plane, at 1/3, 1/2 and 3/4 of the length of that plane.

    Args:
        images (list): list of images to be plotted overlayed (maximum is 3)
    """
    if len(images) > 3:
        raise ValueError("You can only plot a maximum of 3 images overlaid.")

    _, axes = plt.subplots(nrows=3, ncols=3)
    axes = axes.flatten()

    # Define colormap and transparencies of images
    cmaps = ["gray", "hot", "cubehelix"]
    alphas = [1, 0.5, 0.5]

    # Iterate through images
    for i,image in enumerate(images):
        # Load image data and get its shape
        image = nib.load(image)
        data = image.get_fdata()
        shape = np.shape(data)

        # Define slices to plot
        slice_0 = data[shape[0]//3, :, :]
        slice_1 = data[:, shape[1]//3, :]
        slice_2 = data[:, :, shape[2]//3]
        slice_3 = data[shape[0]//2, :, :]
        slice_4 = data[:, shape[1]//2, :]
        slice_5 = data[:, :, shape[2]//2]
        slice_6 = data[shape[0]//4*3, :, :]
        slice_7 = data[:, shape[1]//4*3, :]
        slice_8 = data[:, :, shape[2]//4*3]

        slices=[slice_0, slice_1, slice_2,
                slice_3, slice_4, slice_5,
                slice_6, slice_7, slice_8]

        # Plot each slice and create titles and ylabels
        for j, slice_image in enumerate(slices):
            axes[j].imshow(slice_image.T, cmap=cmaps[i], origin="lower",alpha=alphas[i])
            if j == 0:
                axes[j].set(ylabel="1/3")
                axes[j].set(title="Sagital")
            elif j == 1:
                axes[j].set(title="Coronal")
            elif j == 2:
                axes[j].set(title="Axial")
            elif j == 3:
                axes[j].set(ylabel="1/2")
            elif j == 6:
                axes[j].set(ylabel="3/4")

def force_delete_file(file):
    """This function forces deletes a file with filepath "file" 
    thus it does not prompt anything to the user and does not yield any error.

    Args:
        file (str): path of the file you want to delete
    """
    if os.path.exists(file):
        os.remove(file)

def get_resolution(filename):
    """Get the resolution of an mri image from its path

    Args:
        filename (str): path of the image

    Returns:
        list: resolution of the image (dimensions of the voxel)
    """
    image = nib.load(filename)
    hdr = image.header
    pixdim = hdr["pixdim"]
    return pixdim[1:4]

def calculate_isotropy(resolution):
    """This function calculates the isotropy of an image 
    by calculating the mean of the ratios between each voxel dimension

    Args:
        resolution (list): resolution of the image

    Returns:
        float: isotropy of the image
    """
    ratio_xyz = [resolution[1] / resolution[0],
                 resolution[2] / resolution[0],
                 resolution[2] / resolution[1]]
    return np.mean(ratio_xyz)

def change_datatype(filename, newtype = "uint16"):
    """This function alters the datatype of an image to the specified one

    Args:
        filename (str): path of the image
        newtype (str, optional): data type to change the image to. Defaults to "uint16".
    """
    image = nib.load(filename)
    data = image.get_fdata()
    newimage = nib.Nifti1Image(data.astype(newtype), image.affine)
    nib.save(newimage,filename)

def extract_firstvolume(filename, out_filename=None):
    """extract_firstvolume extracts the first volume of an image

    Args:
        filename (string): filename of the image to be extracted the first volume
        out_filename (filename, optional): File name of the output image file which 
        is the first image but with just the first volume. If no name is given, 
        then it outputs the image in the nibabel.nifti1.Nifti1Image format. Defaults to None.

    Returns:
        new_image: if out_filename is None it will return the image (just the first volume)
        in the nibabel.nifti1.Nifti1Image format. If out_filename is a valid filename, 
        it will save the image with that path, returning None.
    """
    image = nib.load(filename)
    image_data = image.get_fdata()
    image_data2 = image_data[:,:,:,0]
    new_image = nib.nifti1.Nifti1Image(image_data2, image.affine, image.header)
    if out_filename is None:
        return new_image
    else:
        nib.nifti1.save(new_image, out_filename)
        print("Image saved as " + out_filename)
        return None

def register(static, moving, output_image, transform_type="rigid", out_affine=None):
    """register_rigid registers the moving image to the static image 
    using a rigid transformation (center of mass, translation, rigid). 
    It creates (saves) the output image and optionally created the affine 
    matrix of the transformation.

    Args:
        static (string): Filename of the image in the space we want to register to.
        moving (string): Filename of the image to be moved to the static image space.
        output_image (string): Filename of the transformed/registered image.
        out_affine (string, optional): Filename of the affine transformation matrix
        to be created. Defaults to None in which no file is created
        (in fact it is but is is deleted).
    """

    if out_affine is None:
        sub.run(["dipy_align_affine", static, moving, "--transform", transform_type,
                 "--out_moved", output_image,"--force"], check=False)
        force_delete_file("affine.txt")
    else:
        sub.run(["dipy_align_affine", static, moving, "--transform", transform_type,
                "--out_moved", output_image, "--force", "--out_affine", out_affine], check=False)

    change_datatype(output_image)

def apply_transform(static, moving, matrix, output_image):
    """apply_transform applies the transformation matrix to 
    the moving image to register it to the static image space. 
    It creates the transformed image file.

    Args:
        static (string): Filename of the image in the space we want to register to.
        moving (string): Filename of the image to be moved to the static image space.
        matrix (string): Filename of the matrix file to be used for the transformation
        output_image (string): Filename of the transformed image.
    """
    sub.run(["dipy_apply_transform", static, moving, matrix,
             "--out_file", output_image,"--force"], check=False)
    change_datatype(output_image)

def preprocess(image, output, use_fsl=False):
    """This function preprocessess the image by reorienting it to RAS, 
    then if the user wants to, uses robustFOV to crop the fov. 
    Then, bias field correction and gaussian denoising is applied

    Args:
        image (string): input filename of image to apply the preprocessing
        output (string): filename of the image output
        use_fsl (bool, optional): Whether to use FSL functions (requires FSL to be installed).
        Defaults to False.
    """

    if not use_fsl: # Pipeline python
        # Reorient image to RAS
        reorient = Reorient(orientation='RAS')
        reorient.inputs.in_file = image
        res = reorient.run()
        reoriented = res.outputs.out_file

        # Bias field correction
        reoriented_ants = ants.image_read(reoriented)

        # Create mask to prevent intensity normalization in every
        # iteration of the bias field correction
        mask = ants.threshold_image(reoriented_ants,low_thresh=1e-5,
                                    high_thresh=np.inf,inval=1,outval=0,binary=True)

        image_biascorrected = ants.n4_bias_field_correction(reoriented_ants,mask,False)
        #ants.image_write(image_biascorrected, "image_biascorrected.nii.gz")

        # Remove Gaussian Noise
        imagedenoise = ants.denoise_image(image_biascorrected, mask, noise_model = "Gaussian")

        # Write image to file
        ants.image_write(imagedenoise, output)

        # Save as uint16
        change_datatype(output)

        # Delete temporary file
        if os.path.exists(res.outputs.out_file) and res.outputs.out_file!=image:
            os.remove(res.outputs.out_file)

    else: # Pipeline FSL
        sub.run(["fslreorient2std", image, "temp_reoriented.nii.gz"], check=False)
        sub.run(["robustfov", "-i",  "temp_reoriented.nii.gz", "-r",
                 "temp_reoriented_fov.nii.gz"], check=False)
        sub.run(["N4BiasFieldCorrection", "-d", "3", "-i", "temp_reoriented_fov.nii.gz",
                 "-o", "temp_reoriented_fov_bias.nii.gz" ], check=False)
        sub.run(["DenoiseImage", "-d", "3", "-i", "temp_reoriented_fov_bias.nii.gz", "-n",
                 "Gaussian", "-o", "temp_reoriented_fov_bias_denoise.nii.gz"], check=False)
        resampled = ants.resample_image(ants.image_read("temp_reoriented_fov_bias_denoise.nii.gz"),
                                        [240,240,155],True,0)
        ants.image_write(resampled, output)

    # Remove temporary files
    for file in ["CT1_ras.nii.gz", "CT1.mat","T1.mat", "T2.mat", "FLAIR.mat", "temp_reoriented_fov.nii.gz",
                 "temp_reoriented.nii.gz", "temp_reoriented_fov_bias.nii.gz", 
                 "temp_reoriented_fov_bias_denoise.nii.gz"]:
        force_delete_file(os.path.join(os.getcwd(),file))

def create_count_column(table_all, images, name_column):
    """This function will create a new column in the table_all dataframe named name_column that will count how many past images (of the images list) using the logic that each timepoint has in its past (and present)

    Args:
        table_all (dataframe): dataframe that will be modified
        images (list): list of images to count
        name_column (string): name of the new column that will be added
    """

    def have_image(row, list_images):
        """This function returns true if the row in the table has the images in the list, following the AND or OR logic
    Args:
        row (series): Row of the dataframe
        logic (str, optional): Logic to determine if the row has the images. Defaults to "or".
        list_images (list): list of images
    Returns:
        bool: True if the patient has the images with the logic and false otherwise
    """

        found = True
        for image in list_images:
            if not row[image]:
                found=False
        return found

    table_all[name_column] = 0

    # Iterate over each patient
    for patient in table_all['Patient'].unique():
        patient_mask = table_all['Patient'] == patient

    # Iterate over each row for the current patient
        for index, _ in table_all[patient_mask].iloc[::-1].iterrows():
            count = 0
        # Update the count for each row based on the images value
            for index2, row2 in table_all[patient_mask].iloc[::-1].iterrows():
                if index2>index: # Counts with the current image
                    continue
                else:
                    if have_image(row2,images):
                        count += 1
                    else: break
            table_all.at[index, name_column] = count



def remove_timepoints_rano(table_all):
    """This function removes the rows that do not have the basic criteria for RANO classification

    Args:
        table_all (dataframe): table with rows to be removed
    """
    # Remove pre- and post-op classifications and non-existing
    to_delete=["Post-Op", "Pre-Op", False, "Post-Op/PD"]
    table_classifiable=table_all.copy(deep=True)
    for c in to_delete:
        condition = table_classifiable["Rating (according to RANO, PD: Progressive disease, SD: Stable disease, PR: Partial response, CR: Complete response, Pre-Op: Pre-Operative, Post-Op: Post-Operative)"] == c
        table_classifiable.drop(table_classifiable[condition].index,inplace=True)

    # Remove rows where less than 3 months column is true
    condition = table_classifiable['LessThan3Months'] == True
    table_classifiable.drop(table_classifiable[condition].index,inplace=True)

    # remove less than 3 months in the rationale column
    to_delete=["less than 3 months", "Less than 3 months", "Not a timepoint for RANO measurement"]
    for c in to_delete:
        condition = table_classifiable['Rating rationale (CRET: complete resection of the enhancing tumor, PRET: partial resection of the enhancing tumor, T2-Progr.: T2-Progression, L: Lesion)'].str.startswith(c,na=False)
        table_classifiable.drop(table_classifiable[condition].index,inplace=True)
    return table_classifiable

def probs2logits(probabilities, device="cpu"):
    """This function, given a tensor with class probabilities in each row, 
    turns each row into the corresponding logit

    Args:
        probabilities (tensor): Tensor of class probabilities
        device (string): device in which computation is made

    Returns:
        tensor: Tensor of logits
    """
    
    max_indices = torch.argmax(probabilities, dim=1)
    logits=torch.zeros(probabilities.size(), device=device)
    logits[torch.arange(probabilities.size(0)), max_indices] = 1
    return logits

def check_files_in_subdirectories(root_dir, target_files):
    """This function iterates through the folders in the root_dir directory 
    to check if every folder has the file sin target_files

    Args:
        root_dir (str): root directory of the folders you want to check
        target_files (list): list of files to check
    """
    for root, _, files in os.walk(root_dir):
        if root == root_dir:
            continue
        for file in target_files:
            if file not in files:
                print(f"File '{file}' not found in directory: {root}")

def convert2binary(data, labels, dataset):
    """This function converts the labels in the dataset (4 classes) 
    into binary (2 classes) by performing the following assignment:
    0 -> 0
    1 or 2 or 3 ->1

    Args:
        data (_type_): _description_
        labels (list): list of labels of data
        dataset (string): name of the dataset

    Returns:
        list: 
    """
    for i, _ in enumerate(labels):
        if labels[i] != 0:
            labels[i] = 1
            data[i]["label"] = torch.tensor([0, 1], dtype = torch.float32)
        else:
            data[i]["label"] = torch.tensor([1, 0], dtype = torch.float32)
    classes      = ["PD", "other"]
    num_classes  = len(classes)
    new_dataset  = dataset + "_bin"
    return data, labels, classes, num_classes, new_dataset


def get_data_and_transforms(data_dir, table_all, classes, subtract):
    """This function creates the list of timepoints in which 
    each timepoint has the images necessary and the labels

    Args:
        data_dir (string): path to the dataset directory
        table_all (dataframe): dataframe with all the RANO info needed regarding with data to use
        classes (list): list of classes
        subtract (bool): whether to perform subtration between the two timepoints

    Returns:
        list: list with a dictionary for classifiable timepoint, 
        the transforms to be applied to the images, the number of channels 
        to be used by the models and the list of labels of each datapoint

    """
    n_classes = len(classes)
    data = []
    labels= []
    for timepoint in sorted(os.listdir(data_dir)):
        timepoint_dir = os.path.join(data_dir, timepoint)    # Get full directory
        patient, week = timepoint.split('_')                # Get patient and timepoint

        # Get RANO value and turn it into logit
        result = table_all[(table_all['Patient'] == patient) & (table_all['Timepoint'] == week)]
        if patient == "Patient-043" and week == "week-106":
            continue
        else:
            rano = result['RANO'].values[0]
        if rano not in classes:
            print(timepoint) # For sanity check

        # Get label and calculate the logit
        label=classes.index(rano)
        labels.append(label)
        # Turn classes into logits
        logits = func.one_hot(torch.tensor(label), num_classes=n_classes)

        # Create dictionary to store the timepoint's images and its label
        timepoint_dict= {"label": logits}

        # Add modalities present to dictionary
        available_mods = [mod for mod in ["CT1", "T1", "T2", "FLAIR"]
                          if mod+".nii.gz" in os.listdir(timepoint_dir)]
        for modality in available_mods:
            timepoint_dict["image0_" + modality]=os.path.join(timepoint_dir,
                                                              modality + ".nii.gz")
            timepoint_dict["image-1_" + modality]=os.path.join(timepoint_dir,
                                                               modality + "_T-1.nii.gz")

        # Append dictionary to data
        data.append(timepoint_dict)

    # Create transforms to apply to the images
    image_key_list=[key for key in data[0].keys() if key.startswith('image')]

    # Calculate how many channels will be needed according to the dataset in use
    dataset = os.path.basename(data_dir)
    n_modalities=len(dataset.split("_"))-2

    # Define if we want to subtract the images in each modality
    if subtract:
        to_subtract= [["image0_"+mod,"image-1_"+mod] for mod in available_mods]
        to_subtract_names= available_mods
        transforms_train = Compose([
            LoadImaged(keys = image_key_list, ensure_channel_first = True),
            NormalizeIntensityd(keys = image_key_list, channel_wise = True), # Zscore
            ]+[SubtractItemsd(keys = to_subtract[i], name = to_subtract_names[i])
               for i in range(len(to_subtract))]+[
            RandFlipd(keys=to_subtract_names, prob=0.5, spatial_axis=0),
            RandFlipd(keys=to_subtract_names, prob=0.5, spatial_axis=1),
            RandFlipd(keys=to_subtract_names, prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys=to_subtract_names, prob=0.9, factors=0.1, channel_wise=True),
            RandAdjustContrastd(keys=to_subtract_names, prob=0.9),
            RandGaussianNoised(keys=to_subtract_names,prob=0.9),
            ConcatItemsd(keys = to_subtract_names, name = "images"),
            DeleteItemsd(keys = image_key_list+to_subtract_names)])

        transforms_test = Compose([
            LoadImaged(keys = image_key_list, ensure_channel_first = True),
            NormalizeIntensityd(keys = image_key_list, channel_wise = True),
            ]+[SubtractItemsd(keys = to_subtract[i], name = to_subtract_names[i])
               for i in range(len(to_subtract))]+
            [ConcatItemsd(keys = to_subtract_names, name = "images"),
             DeleteItemsd(keys = image_key_list+to_subtract_names)])
        num_channels = n_modalities
    else:
        transforms_train = Compose([
            LoadImaged(keys = image_key_list, ensure_channel_first = True),
            NormalizeIntensityd(keys = image_key_list, channel_wise = True),
            RandFlipd(keys=image_key_list, prob=0.5, spatial_axis=0),
            RandFlipd(keys=image_key_list, prob=0.5, spatial_axis=1),
            RandFlipd(keys=image_key_list, prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys=image_key_list, prob=0.9, factors=0.1, channel_wise=True),
            RandAdjustContrastd(keys=image_key_list, prob=0.9),
            RandGaussianNoised(keys=image_key_list, prob=0.9),
            ConcatItemsd(keys = image_key_list, name = "images"),
            DeleteItemsd(keys = image_key_list)
        ])
        transforms_test = Compose([
            LoadImaged(keys = image_key_list, ensure_channel_first = True),
            NormalizeIntensityd(keys = image_key_list, channel_wise = True),
            ConcatItemsd(keys = image_key_list, name = "images"),
            DeleteItemsd(keys = image_key_list)
        ])
        num_channels = 2 * n_modalities

    return data, transforms_train, transforms_test, num_channels, labels


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
                weights.append(weight)
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


def get_model_setup(model_name, class_prevalence, device, learning_rate, weight_decay, loss_weight, num_channels, num_classes):
    if model_name == "monai_classifier":
        model_config  = Classifier(in_shape = (num_channels, 240, 240, 155), classes = num_classes, channels = (2, 4, 8), strides = (2, 4, 8), last_act = "softmax")
    elif model_name == "monai_densenet121":
        model_config  = DenseNet121(spatial_dims = 3, in_channels = num_channels, out_channels = num_classes, pretrained = False)
    elif model_name == "monai_densenet169":
        model_config  = DenseNet169(spatial_dims = 3, in_channels = num_channels, out_channels = num_classes, pretrained = False)
    elif model_name == "monai_densenet264":
        model_config  = DenseNet264(spatial_dims = 3, in_channels = num_channels, out_channels = num_classes, pretrained = False)
    elif model_name == "monai_vit":
        model_config  = ViT(in_channels = num_channels, img_size = [240, 240, 155], patch_size = [20, 20, 10], classification = True, num_classes = num_classes, pos_embed_type = 'sincos', dropout_rate = 0.1)
    elif model_name == "monai_resnet":
        model_config  = ResNet("bottleneck", [3, 4, 6, 3], [64, 128, 256, 512], spatial_dims = 3, n_input_channels = num_channels, num_classes = num_classes)
    elif model_name == "AlexNet3D":
        model_config  = AlexNet3D(num_channels, num_classes = num_classes)  
    elif model_name == "medicalnet_resnet18":
        from modelresnet import resnet18
        model_config  = resnet18(sample_input_W=240, sample_input_H=240, sample_input_D=155, shortcut_type='A', no_cuda=False, num_seg_classes=4)  
    elif model_name == "densenet264clinical":
        image_model  = DenseNet264(spatial_dims=3, in_channels=num_channels, out_channels=num_classes, pretrained=False)
        model_config = DenseNetWithClinical(densenet_model=image_model, num_classes=num_classes, clinical_data_dim=5)
    else: sys.exit('Please choose one of the models available. You did not write any one of them')

    model     = model_config.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), learning_rate, weight_decay = weight_decay)

    # Define weight of loss function
    if loss_weight == "equal": 
        weight = None
    elif loss_weight == "1/prev":
        weight = torch.tensor([1/a for a in class_prevalence], dtype = torch.float32).to(device)
    elif loss_weight=="1-prev":
        weight = torch.tensor([1-a for a in class_prevalence], dtype = torch.float32).to(device)
    else:
        sys.exit('Please choose one of the weighting types available.')

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



def create_tensorboard(n_epochs, bs, learning_rate, logs_folder,
                       device, model_name, dataset, subtract, weight_decay,
                       weight, loss_function, stop_decrease, decrease_LR, sampler_weight, dec_LR_factor, fold):
    
    """This function generates a tensorboard folder to keep the evolution of the model training

    Args:
        n_epochs (int): number of (maximum) epochs
        bs (int): batch size
        learning_rate (float): learning rate for the optimizer
        logs_folder (str): folder where the logs will be saved
        device (device): name of device
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
    log_dir = os.path.join(logs_folder, current_time)
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



def train(log_dir, writer, train_loader, test_loader, model_name, dataset,
          device, learning_rate, n_epochs, optimizer, seed, weight_decay, 
          loss_function, model, decrease_LR, dec_LR_factor, patience, clinical_data):
    """Training and validation loop

    Args:
        log_dir (str): log directory
        writer (obj): writer object of tensorflow
        train_loader (dataloader): dataloader of training data
        test_loader (dataloader): dataloader of test data
        model_name (str): name of the model to use
        device (device): name of device
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

    torch.save(model.state_dict(), os.path.join(log_dir, "checkpoint" + ".pt"))
    os.chmod(log_dir,0o777)
    print("Starting training...")
    # Run learning
    torch.cuda.empty_cache()
    for epoch in range(n_epochs):
        # Early stopping if lr decreases more then patience times
        if lr_decreases == patience:
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
            inputs, labels = batch_data["images"].to(device), batch_data["label"].float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs) if not clinical_data else model(inputs,batch_data["clinical"].to(device))
            loss = loss_function(outputs, labels) if model_name != "monai_vit" else loss_function(outputs[0], labels)
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
                val_images, val_labels = val_data["images"].to(device), val_data["label"].float().to(device)
                outputs = model(val_images) if not clinical_data else model(val_images, val_data["clinical"].to(device))
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

            print("current epoch: {}; current accuracy: {:.4f}; current AUC: {:.4f}; best loss: {:.4f}, with acc= {:.4f} at epoch {}".format(
                epoch + 1, acc_value, auc_result, best_metric, best_acc, best_metric_epoch))

            # Write to writer
            writer.add_scalar("val_accuracy", acc_value, epoch + 1)
            writer.add_scalar("AUC", auc_result, epoch + 1)
            writer.add_scalar("val_loss", val_loss, epoch + 1)

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    os.remove(os.path.join(log_dir, "checkpoint" + ".pt"))
    writer.close()

def critical_error(y_pred, y):
    """This function calculated the probability of making  crucial error. 
    A crucial error is either when the function predicts disease development (0 or 1) 
    but is is actually response (2 or 3), or when it predicts response development but 
    it is actually disease development

    Args:
        y_pred (tensor or list): Class prediction of the model. It is either a 
        list or a tensor with the numbers corresponding to the predicted class of the model
        y (tensor or list): Real value of classes. It is either a list or a 
        tensor with the numbers corresponding to the real classes of the inputs.

    Returns:
        float: percentage of critical errors (0-1)
    """
    # Turn tensors into lists
    if torch.is_tensor(y_pred):
        y_pred = y_pred.tolist()
    if torch.is_tensor(y):
        y = y.tolist()

    # Vectors must be the same length
    assert len(y) == len(y_pred), "y_pred and y must be the same length"

    # Count number of critical errors
    count_errors = 0
    for i,_ in enumerate(y):
        value_y = y[i] + 1
        value_ypred = y_pred[i] + 1

        if value_y in (1, 2):
            value_y *= -1
        if value_ypred in (1, 2):
            value_ypred *= -1

        if value_ypred * value_y < 0:
            count_errors += 1
    return count_errors/len(y)

def test(model_config, log_dir, dataset, device, bs, classes, test_loader, model_name, clinical_data):
    """Function to test model

    Args:
        model_config (_type_): _description_
        log_dir (_type_): _description_
        dataset (_type_): _description_
        device (_type_): _description_
        bs (_type_): _description_
        classes (_type_): _description_
        test_loader (_type_): _description_
        model_name (_type_): _description_

    Returns:
        _type_: _description_
    """
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
            test_images, test_labels = batch["images"].to(device), batch["label"].float().to(device)
            real = torch.cat([real, test_labels.argmax(dim = 1)])
            real_onehot = torch.cat([real_onehot, test_labels], dim = 0)

            # Get output/predicted values
            outputs = model_trained(test_images) if not clinical_data else model_trained(test_images, batch["clinical"].to(device))
            outputs_probs = sigsoft(outputs) if model_name != "monai_vit" else sigsoft(outputs[0])
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




def plot_image(image, norm = None):
    """This function plots the axial slice of an image along with a z slider so that the z coordinate can be changed dynamically

    Args:
        image (string): file path of the image to plot
        norm (None or string): see norm parameter of plt.imshow
    """

    # Load data of image
    if type(image) is str:
        img = nib.load(image)
        data = img.get_fdata()
    elif type(image) is MetaTensor:
        data = image.cpu().numpy()
    else:
        data=image

    # Function to display axial slice
    def display_axial_slice(z):
        max_image=np.max(np.max(np.max(data)))
        if norm=="log":
            min_image=np.min(np.min(np.min(data))) if np.min(np.min(np.min(data))) !=0 else 0.0000001
        else:
            min_image=np.min(np.min(np.min(data)))
        plt.figure(figsize = (6, 6))
        plt.imshow(data[:, :, z].T, cmap = 'gray', origin = 'lower',vmin=min_image,vmax=max_image, norm=norm)
        plt.title(f'Axial Slice at z={z}')
        plt.colorbar()
        plt.show()
        

    # Interactively select axial slice using a slider
    z_slider = IntSlider(min = 0, max = data.shape[2] - 1, step = 1, value = data.shape[2] // 2, orientation = 'vertical', description = 'Axial slice')
    w = interactive(display_axial_slice, z=z_slider)

    box_layout = Layout(align_items = 'center')

    display(HBox([w.children[1], w.children[0]], layout = box_layout))

def plot_saliency(image_data, saliency, overlay=False, cmap="gray"):
    """This function plots the saliency map along with the input image (data) of a certain prediction.

    Args:
        image (numpy array): image data to be plotted
        saliency (numpy array): saliency map to be plotted
    """

    # Function to display axial slice
    def display_axial_slice_overlay(z):
        max_saliency=np.max(np.max(np.max(saliency)))
        min_saliency=np.min(np.min(np.min(saliency)))
        max_image=np.max(np.max(np.max(image_data)))
        min_image=np.min(np.min(np.min(image_data)))

        plt.figure(figsize = (6, 6))
        plt.imshow(image_data[:, :, z].T, cmap = 'gray', origin = 'lower',vmin=min_image, vmax=max_image)
        plt.imshow(saliency[:, :, z].T, cmap = cmap, origin = 'lower',alpha=0.5,vmin=min_saliency,vmax=max_saliency)
        plt.xticks(list(np.linspace(0, image_data.shape[0], 11)))
        plt.yticks(list(np.linspace(0, image_data.shape[1], 11)))
        plt.title(f'Axial Slice at z={z}')
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    def display_axial_slice(z):
        max_saliency=np.max(np.max(np.max(saliency)))
        min_saliency=np.min(np.min(np.min(saliency)))
        max_image=np.max(np.max(np.max(image_data)))
        min_image=np.min(np.min(np.min(image_data)))

        plt.figure(figsize = (6, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image_data[:, :, z].T, cmap = 'gray', origin = 'lower',vmin=min_image,vmax=max_image)
        plt.title(f'Axial Slice at z={z}')
        plt.xticks(list(np.linspace(0, image_data.shape[0], 11)))
        plt.yticks(list(np.linspace(0, image_data.shape[1], 11)))
        plt.colorbar()
        plt.subplot(1,2,2)
        plt.imshow(saliency[:, :, z].T, cmap = 'jet', origin = 'lower',vmin=min_saliency,vmax=max_saliency)
        plt.title(f'Axial Slice at z={z}')
        plt.xticks(list(np.linspace(0, image_data.shape[0], 11)))
        plt.yticks(list(np.linspace(0, image_data.shape[1], 11)))
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    # Interactively select axial slice using a slider
    z_slider = IntSlider(min = 0, max = image_data.shape[2] - 1, step = 1, value = image_data.shape[2] // 2, orientation = 'vertical', description = 'Axial slice')
    if overlay:
        w = interactive(display_axial_slice_overlay, z = z_slider)
    else:
        w = interactive(display_axial_slice, z = z_slider)
    box_layout = Layout(align_items = 'center')

    display(HBox([w.children[1], w.children[0]], layout = box_layout))

def plot_saliency_grid(image_data, saliency, overlay=False, cmap='gray', num_rows=3, num_columns=5, filename=None):
    """This function plots the saliency map along with the input image (data) in a 3x5 grid.
    
    Args:
        image_data (numpy array): image data to be plotted
        saliency (numpy array): saliency map to be plotted
        overlay (bool): whether to overlay saliency on image
        num_slices (int): number of slices to show (default 15 for 3x5 grid)
    """
    num_slices=num_rows*num_columns

    # Calculate slice indices to show
    z_indices = np.linspace(0, image_data.shape[2]-1, num_slices, dtype=int)
    
    # Calculate global min/max values
    max_saliency = np.max(saliency)
    min_saliency = np.min(saliency)
    max_image = np.max(image_data)
    min_image = np.min(image_data)

    for idx, z in enumerate(z_indices):
        if overlay:      
            # Plot overlaid images
            plt.imshow(image_data[:, :, z].T, cmap='gray', origin='lower', vmin=min_image, vmax=max_image)
            plt.imshow(saliency[:, :, z].T, cmap=cmap, origin='lower', alpha=0.5, vmin=min_saliency, vmax=max_saliency)
            plt.title(f'z={z}')

        else:
            # Create side-by-side plots within each grid cell
            plt.subplot(num_rows, num_columns*2, 2*idx + 1)
            plt.imshow(image_data[:, :, z].T, cmap='gray', origin='lower', vmin=min_image, vmax=max_image)
            plt.title(f'z={z}')
            plt.axis('off')
            
            plt.subplot(num_rows, num_columns*2, 2*idx + 2)
            plt.imshow(saliency[:, :, z].T, cmap=cmap, origin='lower', vmin=min_saliency, vmax=max_saliency)
            plt.title(f'Saliency z={z}')
            plt.axis('off')
    
    plt.suptitle('Axial Slices with Saliency Maps', fontsize=16)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, format="svg", dpi=500)
    plt.show()

def make_gif_saliency(image_data, saliency,name):
    frames = []
    max_image=np.max(np.max(np.max(saliency)))
    min_image=np.min(np.min(np.min(saliency)))
    for z in range(image_data.shape[-1]):
        plt.imshow(image_data[:, :, z].T, cmap = 'gray', origin = 'lower')
        plt.imshow(saliency[:, :, z].T, cmap = 'jet', origin = 'lower',alpha=0.5,vmin=min_image,vmax=max_image)
        plt.xticks(list(np.linspace(0, image_data.shape[0], 11)))
        plt.yticks(list(np.linspace(0, image_data.shape[1], 11)))
        plt.title(f'Axial Slice at z={z}')
        plt.colorbar()
        plt.tight_layout()
        os.makedirs("./temp", exist_ok=True)
        plt.savefig(f'./temp/frame_{z}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        frames.append(imageio.imread(f'./temp/frame_{z}.png'))
        print(str(z)+"/"+str(image_data.shape[-1]))
    os.makedirs("./images", exist_ok=True)
    output_path = './images/'+name+'.gif'

    imageio.mimsave(output_path, frames, duration=5,loop=0)  # Duration between frames in seconds
    shutil.rmtree("./temp")
