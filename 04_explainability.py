#%% 
# Imports
print("Importing...")
import os
import pickle
import resource
import sys

import matplotlib.pyplot as plt
import monai
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as func
from captum.attr import Saliency
from monai.data import DataLoader, Dataset
from monai.networks.nets import (DenseNet121, DenseNet169,
                                 DenseNet264, ViT)
from monai.utils import set_determinism
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from models import (AlexNet3D, DenseNetWithClinical)
from utils import get_data_and_transforms, plot_saliency_grid

cuda=torch.cuda.is_available()

torch.cuda.empty_cache()
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

#%% Variables that can change
phd_dir        = os.getcwd()
classes        = ["PD", "SD", "PR", "CR"]
num_classes    = len(classes)
logs_folder    = "./Pytorch_Logs"
bs             = 1
seed           = 2
set_determinism(seed)

# CHANGE FOR BEST EXPERIMENT
experiment     = "24-10-18_14.29.31"
fold           = 1

df = pd.read_csv('results.csv', header=None)
row = df[df[0] == "Jul09_15-31-18_pluto"]
experiment_list = row.iloc[0]

dataset        = "rano" + experiment_list[9]
data_dir       = os.path.join("LUMIERE/Datasets/", dataset)
subtract       = experiment_list[11]==True
clinical_data  = experiment_list[-1]==True
model_name     = experiment_list[15] # densenet264clinical monai_vit monai_resnet monai_classifier monai_densenet121 monai_densenet169 monai_densenet264 monai_densenet201 jang_model_mri AlexNet3D GoogleNet3D

log_dir        = os.path.join(logs_folder,experiment)

#%% Create dataset to classify image types 

print("Getting data...")

# Get table with data info
with open(os.path.join(phd_dir,"table_classifiable.pkl"), "rb") as f:
    table_all = pickle.load(f)

# Create the data, get transforms and number of channels
data,  _, transforms_test, num_channels, labels = get_data_and_transforms(data_dir, table_all, classes, subtract, clinical_data)

#%% Divide data into train and test set and create data loader

print("Creating train and validation datasets...")

# Recreate fold from benchmark
folds = monai.data.utils.partition_dataset_classes(data, labels, num_partitions = 5, seed = seed)
test_data = folds[fold]

# Create datasets
test_ds  = Dataset(data = test_data, transform = transforms_test)
test_loader  = DataLoader(test_ds, batch_size = bs, shuffle = True)


# %%
# Create DL model
device = "cpu"

#################
# Model Options #
#################
if model_name == "monai_densenet121":
    model_config  = DenseNet121(spatial_dims=3, in_channels=num_channels, out_channels=num_classes, pretrained=False)
elif model_name == "monai_densenet169":
    model_config  = DenseNet169(spatial_dims=3, in_channels=num_channels, out_channels=num_classes, pretrained=False)
elif model_name == "monai_densenet264":
    model_config  = DenseNet264(spatial_dims=3, in_channels=num_channels, out_channels=num_classes, pretrained=False)
elif model_name == "monai_vit":
    model_config  = ViT(in_channels=num_channels, img_size=[240,240,155], patch_size=[20,20,10], classification=True, num_classes=num_classes, pos_embed_type='sincos', dropout_rate=0.1)
elif model_name == "AlexNet3D":
    model_config  = AlexNet3D(num_channels, num_classes = num_classes)  
elif model_name == "medicalnet_resnet18":
    from models import resnet18
    model_config  = resnet18(sample_input_W=240, sample_input_H=240, sample_input_D=155, shortcut_type='A', no_cuda=False, num_seg_classes=4)  
elif model_name == "densenet264clinical":
    image_model  = DenseNet264(spatial_dims=3, in_channels=num_channels, out_channels=num_classes, pretrained=False)
    model_config = DenseNetWithClinical(densenet_model=image_model, num_classes=num_classes, clinical_data_dim=5)
else: sys.exit('Please choose one of the models available. You did not write any one of them')
print("Model imported")

# Load the pre-trained model
model = model_config.to(device)
weights_path = os.path.join(log_dir,dataset + ".pt")
model.load_state_dict(torch.load(weights_path, map_location = device),strict=False)
model.eval()
print("Weights loaded")

it = iter(test_loader) # Create an input tensor image for your model.
input_item = next(it)
input_tensor = input_item["images"]
print("batch loaded")
print(input_item["label"])
#%% Select class (do not run for first class and then from this section on for the other classes)

while not torch.equal(input_item["label"],torch.tensor([[0,1,0,0]])): # change tensor for the classes 2 and 3
    input_item = next(it)
    input_tensor = input_item["images"]
    print("batch loaded")
    print(input_item["label"])
# %% GRAD-CAM

outputs=model(input_tensor.to(device))
class_pred = int(torch.max(outputs))
class_real = int(torch.argmax(input_item["label"]))

target_real = [ClassifierOutputTarget(class_real)]
target_pred = [ClassifierOutputTarget(class_pred)]

target_layers = [model.features[10]] 

torch.cuda.empty_cache()
# Construct the CAM object
with GradCAM(model=model, target_layers=target_layers) as cam:
    grayscale_cam_real = cam(input_tensor=input_tensor, targets = target_real)
    grayscale_cam_pred = cam(input_tensor=input_tensor, targets = target_pred)

grayscale_cam_real = grayscale_cam_real[0, :]
grayscale_cam_pred = grayscale_cam_pred[0, :]

plot_saliency_grid(input_tensor[0,-3,:,:,:],grayscale_cam_real,
                    overlay=True, cmap="jet", filename="images/final/grad-cam_c"+str(class_real)+".svg")
plot_saliency_grid(input_tensor[0,-3,:,:,:],grayscale_cam_pred, 
                    overlay=True, cmap="jet", filename="images/final/grad-cam_c"+str(class_real)+"_pred.svg")

# %% Captum

_, predicted = torch.max(outputs, 1)
input_tensor.requires_grad=True
print('Ground truth:', classes[input_item["label"].argmax()], "\n",
    'Predicted:', classes[predicted], "\n",
    'Probability:', torch.max(func.softmax(outputs, 1)).item(),"\n",
    'Probability for all classes:', np.round(func.softmax(outputs, 1).tolist()[0],3)
    )

model.to("cpu")
images=input_tensor.to("cpu").squeeze()
channel=-3
original_image =images.cpu().detach().numpy()
original_image =np.flip(np.transpose(original_image, (0, 2, 1, 3)),1)

saliency = Saliency(model)

grads_pred = saliency.attribute(input_tensor, target=class_pred)
grads_real = saliency.attribute(input_tensor, target=class_real)

slice_list=[[55,77,99], [55,77,99], [66,77,88], [44,66,77]]

for SLICE in slice_list[class_real]:

    # Plot saliency maps for real class
    max_image=np.max(np.max(np.max(grads_real)))
    min_image=np.min(np.min(np.min(grads_real))) if np.min(np.min(np.min(grads_real))) !=0 else 0.0000001
    plt.figure(figsize = (6, 6))
    plt.imshow(grads_real[0,channel,:, :, SLICE].T, cmap = 'gray', origin = 'lower',vmin=min_image,vmax=max_image, norm="log")
    plt.title(f'Axial Slice at z={SLICE}')
    plt.colorbar()
    plt.savefig("images/final/saliency_c"+str(class_real)+"_s"+str(SLICE)+".svg", format="svg", dpi=500)
    plt.show()

    # Plot saliency maps for predicted class
    max_image=np.max(np.max(np.max(grads_pred)))
    min_image=np.min(np.min(np.min(grads_pred))) if np.min(np.min(np.min(grads_pred))) !=0 else 0.0000001
    plt.figure(figsize = (6, 6))
    plt.imshow(grads_pred[0,channel,:, :, SLICE].T, cmap = 'gray', origin = 'lower',vmin=min_image,vmax=max_image, norm="log")
    plt.title(f'Axial Slice at z={SLICE}')
    plt.colorbar()
    plt.savefig("images/final/saliency_c"+str(class_real)+"_s"+str(SLICE)+"_pred.svg", format="svg", dpi=500)
    plt.show()
