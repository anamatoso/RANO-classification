#%% Imports

import os
import pickle
import subprocess as sub

import pandas as pd

from utils import check_files_in_subdirectories, create_count_column, remove_timepoints_rano

#%% Load data - create table all

# Import CSVs
table_RANO = pd.read_csv("./lumiere/csvs/LUMIERE-ExpertRating_summary.csv",
                         delimiter = ",",header = 0)
table_completeness = pd.read_csv("./lumiere/csvs/LUMIERE-datacompleteness.csv",
                                 delimiter = ",",header = 0)
table_all = table_RANO.merge(table_completeness,on = ["Patient","Timepoint"],how = "outer")
del table_RANO, table_completeness

# Replace crosses and empty with true/false
for column in ["LessThan3Months","RANO","NonMeasurableLesions","RatingRationale","CT1",
               "T1", "T2","FLAIR","DeepBraTumIA","HD-GLIO-AUTO",
               "DeepBraTumIA-CoLlAGe","HD-GLIO-AUTO-CoLlAGe"]:  
    table_all[column] = table_all[column].replace({'x': True, '': False, 'NaN': False, None: False})
del column

# Sort table and reset index
table_all.sort_values(by=['Patient', 'Timepoint'], inplace=True)
table_all.reset_index(drop=True, inplace=True)

# Create columns for knowing how many past images each timepoint 
# has (agreggated by groups of images)
for images in [["CT1"],["CT1","FLAIR"],["CT1","T1","T2","FLAIR"],
               ["T1","T2","FLAIR"],["T1","FLAIR"]]:
    name = "_".join(images) + "_count"
    create_count_column(table_all, images, name)
del name, images

# Save table
save=input("Do you want to save a new table_all [y/n]: ")
if save=="y":
    with open("/home/amatoso/phd/lumiere/table_all.pkl","wb") as f:
        pickle.dump(table_all,f)

# Load table
with open("/home/amatoso/phd/lumiere/table_all.pkl", "rb") as f:
    table_all = pickle.load(f)

DATA_DIR = "/home/amatoso/phd/lumiere/data"

del f
#%% Create classifyable table
images_to_count = ["CT1", "T1", "T2", "FLAIR"]
for images_to_count in [["CT1", "T1", "T2", "FLAIR"],
                        ["CT1", "FLAIR"],["T1", "T2", "FLAIR"],["T1", "FLAIR"],["CT1"]]:
    mods_to_count = "_".join(images_to_count)

    table_classifyable = remove_timepoints_rano(table_all) # remove less than 3 months and pre and post op

    # Create has past column to remove the timepoints that have less timepoints in the past than what we want
    column = mods_to_count + "_count"
    table_classifyable['has_past'] = table_classifyable[column] > 1 # How many past images we want

    table_classifyable.drop((table_classifyable[table_classifyable['has_past'] is False]).index,inplace = True)

    ## Create folder for links to usable data

    # one folder per classifyable timepoint with past image(s)
    sub.call(["rm", "-rf","./lumiere/datasets/rano_" + mods_to_count + "_T-1"])
    count = 0
    for ind in table_classifyable.index[::-1]: #this index is the same as in table all
        count += 1
        path = os.path.join("./lumiere/datasets/rano_" + mods_to_count + "_T-1",
                            table_classifyable["Patient"][ind] + "_" + table_classifyable["Timepoint"][ind])
        os.makedirs(path, exist_ok = True)

        for mod in images_to_count:
            # Add images of current timepoint
            image = os.path.join(DATA_DIR, table_classifyable["Patient"][ind],
                                 table_classifyable["Timepoint"][ind], mod + "_tostd.nii.gz")
            if os.path.exists(image) and not os.path.exists(os.path.join(path, mod + ".nii.gz")):
                os.symlink(image, os.path.join(path, mod + ".nii.gz"))

            # Add images from timepoints from before

            image = os.path.join(DATA_DIR, table_classifyable["Patient"][ind],
                                 table_all["Timepoint"][ind-1], mod + "_tostd.nii.gz")
            if os.path.exists(image) and not os.path.exists(os.path.join(path,
                                                                         mod + "_T-1.nii.gz")):
                os.symlink(image, os.path.join(path, mod + "_T-1.nii.gz"))

    print("Total timepoints in " + mods_to_count + ": " + str(count))

    # Sanity check to see if target files are in directories
    file_list = [im+".nii.gz" for im in images_to_count]+[im+"_T-1.nii.gz" \
                                                          for im in images_to_count]
    check_files_in_subdirectories("/home/amatoso/phd/lumiere/datasets/rano_"+mods_to_count+"_T-1",
                                  file_list)
