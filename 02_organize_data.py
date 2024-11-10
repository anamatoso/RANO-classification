#%% Imports

import os
import pickle
import subprocess as sub

import pandas as pd

from utils import (check_files_in_subdirectories, create_count_column,
                   remove_timepoints_rano)

DATA_DIR = "./LUMIERE/Imaging"
DATASETS =  [["CT1"],["CT1","FLAIR"],["CT1","T1","T2","FLAIR"], 
             ["T1","T2","FLAIR"],["T1","FLAIR"]]
#%% Load data - create table all

# Import CSVs
TABLE_RANO = pd.read_csv("./LUMIERE/LUMIERE-ExpertRating-v202211.csv",
                         delimiter = ",",header = 0)
TABLE_COMPLETENESS = pd.read_csv("./LUMIERE/LUMIERE-datacompleteness.csv",
                                 delimiter = ",",header = 0)
TABLE_ALL = TABLE_RANO.merge(TABLE_COMPLETENESS,on = ["Patient", "Timepoint"],how = "outer")
del TABLE_RANO, TABLE_COMPLETENESS

# Replace crosses and empty with true/false
for column in ["LessThan3Months","Rating (according to RANO, PD: Progressive disease, SD: Stable disease, PR: Partial response, CR: Complete response, Pre-Op: Pre-Operative, Post-Op: Post-Operative)",
               "NonMeasurableLesions","Rating rationale (CRET: complete resection of the enhancing tumor, PRET: partial resection of the enhancing tumor, T2-Progr.: T2-Progression, L: Lesion)",
                "CT1", "T1", "T2", "FLAIR", "DeepBraTumIA", "HD-GLIO-AUTO",
               "DeepBraTumIA-CoLlAGe","HD-GLIO-AUTO-CoLlAGe"]:  
    TABLE_ALL[column] = TABLE_ALL[column].replace({'x': True, '': False, 'NaN': False, None: False})
del column

# Sort table and reset index
TABLE_ALL.sort_values(by=['Patient', 'Timepoint'], inplace=True)
TABLE_ALL.reset_index(drop=True, inplace=True)

# Create columns for knowing how many past images each timepoint 
# has (agreggated by groups of images)
for images in DATASETS:
    name = "_".join(images) + "_count"
    create_count_column(TABLE_ALL, images, name)
del name, images

# Save table with classifyable timepoints
CLASSIFYABLE = TABLE_ALL[
    (TABLE_ALL['LessThan3Months'] == False) & # LessThan3Months must be False
    (TABLE_ALL['RANO'] != "Pre-Op") &         # RANO must not be "Pre-Op"
    (TABLE_ALL['RANO'] != "Post-Op") &        # RANO must not be "Post-Op"
    (TABLE_ALL['RANO'] is not False) &        # RANO must not be False
    (TABLE_ALL['RANO'] != "Post-Op/PD") &     # RANO must not be "Post-Op/PD"
    (~TABLE_ALL['RatingRationale'].str.contains("RANO", na=False)) # RatingRationale must not contain "RANO"
]
CLASSIFYABLE.to_pickle('./table_classifyable.pkl')

#%% Create datasets with classifyable data

DATASETS_DIR = "./Datasets"

for images_to_count in DATASETS:
    mods_to_count = "_".join(images_to_count)
    print("Doing "+mods_to_count+ " dataset")
    table_classifyable = remove_timepoints_rano(TABLE_ALL) # remove less than 3 months and pre and post op

    # Create has past column to remove the timepoints that have less timepoints in the past than what we want
    column = mods_to_count + "_count"
    table_classifyable['has_past'] = table_classifyable[column] > 1 # How many past images we want

    table_classifyable.drop((table_classifyable[table_classifyable['has_past'] == False]).index,inplace = True)

    ## Create folder for links to usable data

    # one folder per classifyable timepoint with past image(s)
    sub.call(["rm", "-rf", os.path.join(DATASETS_DIR, "rano_" + mods_to_count + "_T-1")])
    count = 0
    for ind in table_classifyable.index[::-1]: #this index is the same as in table all
        count += 1
        path = os.path.join(DATASETS_DIR, "rano_" + mods_to_count + "_T-1",
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
                                 TABLE_ALL["Timepoint"][ind-1], mod + "_tostd.nii.gz")
            if os.path.exists(image) and not os.path.exists(os.path.join(path,
                                                                         mod + "_T-1.nii.gz")):
                os.symlink(image, os.path.join(path, mod + "_T-1.nii.gz"))

    print("Total timepoints in " + mods_to_count + ": " + str(count))

    # Sanity check to see if target files are in directories
    file_list = [im+".nii.gz" for im in images_to_count]+[im+"_T-1.nii.gz" \
                                                          for im in images_to_count]
    check_files_in_subdirectories(os.path.join(DATASETS_DIR, "rano_"+mods_to_count+"_T-1"),
                                  file_list)
