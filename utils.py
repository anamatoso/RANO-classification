""" Auxilliary file with utils functions
"""

import os
import subprocess as sub

import ants
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from IPython.display import display
from ipywidgets import HBox, IntSlider, Layout, interactive
from monai.data.meta_tensor import MetaTensor
from monai.transforms import (Compose, ConcatItemsd, DeleteItemsd, LoadImaged,
                              NormalizeIntensityd, RandAdjustContrastd,
                              RandFlipd, RandGaussianNoised,
                              RandScaleIntensityd, SubtractItemsd)
from nipype.interfaces.image import Reorient


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
        type (str, optional): data type to change the image to. Defaults to "uint16".
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
        to be created. Defaults to None in which no file is created (in fact it is but is is deleted).
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
    for file in ["CT1.mat", "T1.mat", "T2.mat", "FLAIR.mat", "temp_reoriented_fov.nii.gz",
                 "temp_reoriented.nii.gz", "temp_reoriented_fov_bias.nii.gz", 
                 "temp_reoriented_fov_bias_denoise.nii.gz"]:
        force_delete_file(file)

def remove_timepoints_rano(table_all):
    """This function removes the rows that dont have the basic criteria for RANO classification

    Args:
        table_all (dataframe): table with rows to be removed
        table_classifyable (dataframe): final table
    """
    # Remove pre and post op classifications and non existing
    to_delete=["Post-Op", "Pre-Op", False, "Post-Op/PD"]
    table_classifyable=table_all.copy(deep=True)
    for c in to_delete:
        condition = table_classifyable['RANO'] == c
        table_classifyable.drop(table_classifyable[condition].index,inplace=True)

    # Remove rows where less than 3 months column is true
    condition = table_classifyable['LessThan3Months'] is True
    table_classifyable.drop(table_classifyable[condition].index,inplace=True)

    # remove less than 3 months in the rationale column
    to_delete=["less than 3 months", "Less than 3 months", "Not a timepoint for RANO measurement"]
    for c in to_delete:
        condition = table_classifyable['RatingRationale'].str.startswith(c,na=False)
        table_classifyable.drop(table_classifyable[condition].index,inplace=True)
    return table_classifyable

def probs2logits(probs, device="cpu"):
    """This function, given a tensor with class probabilities in each row, 
    turns each row into the corresponding logit

    Args:
        probs (tensor): Tensor of class probabilities

    Returns:
        tensor: Tensor of logits
    """
    max_indices = torch.argmax(probs, dim=1)
    logits=torch.zeros(probs.size(),device=device)
    logits[torch.arange(probs.size(0)), max_indices] = 1
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

def get_data_and_transforms(data_dir, table_all, classes, subtract):
    """This function creates the list of timepoints in which 
    each timepoint has the images necessary and the labels

    Args:
        data_dir (string): path to the dataset directory
        table_all (dataframe): dataframe with all the RANO info needed regarding with data to use
        classes (list): list of classes

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
        logits = F.one_hot(torch.tensor(label), num_classes=n_classes)

        # Create dictionary to store the timepoint's images and its label
        timepoint_dict={}
        timepoint_dict["label"]=logits

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

def plot_image(image):
    """This function plots the axial slice of an image along with a z slider 
    so that the z coordinate can be changed dynamically

    Args:
        image (string): file path of the image to plot
    """
    # Load data of image
    if isinstance(image, str):
        img = nib.load(image)
        data = img.get_fdata()
    elif isinstance(image, MetaTensor):
        data = image.cpu().numpy()
    else:
        data=image

    # Function to display axial slice
    def display_axial_slice(z):
        plt.figure(figsize = (6, 6))
        plt.imshow(data[:, :, z].T, cmap = 'gray', origin = 'lower')
        plt.title(f'Axial Slice at z={z}')
        plt.colorbar()
        plt.show()

    # Interactively select axial slice using a slider
    z_slider = IntSlider(min = 0, max = data.shape[2] - 1, step = 1, value = data.shape[2] // 2,
                         orientation = 'vertical', description = 'Axial slice')
    w = interactive(display_axial_slice, z=z_slider)

    box_layout = Layout(align_items = 'center')

    display(HBox([w.children[1], w.children[0]], layout = box_layout))

def plot_saliency(image_data, saliency):
    """This function plots the saliency map along with the input image (data) 
    of a certain prediction.

    Args:
        image (numpy array): image data to be plotted
        saliency (numpy array): saliency map to be plotted
    """
    # Function to display axial slice
    def display_axial_slice(z):
        plt.figure(figsize = (6, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image_data[:, :, z].T, cmap = 'gray', origin = 'lower')
        plt.title(f'Axial Slice at z={z}')
        plt.xticks(list(np.linspace(0, image_data.shape[0], 11)))
        plt.yticks(list(np.linspace(0, image_data.shape[1], 11)))
        plt.colorbar()
        plt.subplot(1,2,2)
        plt.imshow(saliency[:, :, z].T, cmap = 'rainbow', origin = 'lower')
        plt.title(f'Axial Slice at z={z}')
        plt.xticks(list(np.linspace(0, image_data.shape[0], 11)))
        plt.yticks(list(np.linspace(0, image_data.shape[1], 11)))
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    # Interactively select axial slice using a slider
    z_slider = IntSlider(min = 0, max = image_data.shape[2] - 1, step = 1,
                         value = image_data.shape[2] // 2,
                         orientation = 'vertical', description = 'Axial slice')
    w = interactive(display_axial_slice, z = z_slider)
    box_layout = Layout(align_items = 'center')
    display(HBox([w.children[1], w.children[0]], layout = box_layout))
