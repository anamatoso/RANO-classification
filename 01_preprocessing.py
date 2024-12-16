""" Preprocessing file

"""
# Imports
import os
import sys

from utils import (apply_transform, calculate_isotropy, extract_firstvolume,
                   force_delete_file, get_resolution, preprocess, register)

# Define folders
ATLAS_FOLDER    = "./atlases"
DATA_DIR        = "./LUMIERE/Imaging"
MODALITIES      = ["CT1","T1","T2","FLAIR"]

#%%
# Extract first volume of SRI atlases (T1 and T2)
if not os.path.exists(os.path.join(ATLAS_FOLDER,"SRI24_T1_brain.nii")) or not os.path.exists(os.path.join(ATLAS_FOLDER,"SRI24_T2_brain.nii")):
    sys.exit("The T1 and T2 atlases must exist in the atlas folder.")
extract_firstvolume(os.path.join(ATLAS_FOLDER,"SRI24_T1_brain.nii"),
                    os.path.join(ATLAS_FOLDER,"sri24_T1_3D.nii.gz"))
print("First volume of T1 atlas extracted.")

extract_firstvolume(os.path.join(ATLAS_FOLDER,"SRI24_T2_brain.nii"),
                    os.path.join(ATLAS_FOLDER,"sri24_T2_3D.nii.gz"))
print("First volume of T1 atlas extracted.")

for patient in [p for p in sorted(os.listdir(DATA_DIR))]:
    if not os.path.isdir(os.path.join(DATA_DIR,patient)):
        print(patient+" is not a directory, so it will not be processed")
        continue
    p_dir = os.path.join(DATA_DIR,patient)
    for week in [p for p in sorted(os.listdir(p_dir))]:
        if not os.path.isdir(os.path.join(p_dir,week)):
            print(patient+" is not a directory, so it will not be processed")
            continue
        print("Processing "+patient+", "+week)
        w_dir = os.path.join(p_dir,week)

        # List the modalities the time point has and their resolutions
        available_mods = [item for item in MODALITIES if item+".nii.gz" in os.listdir(w_dir)]
        resolutions=[get_resolution(os.path.join(w_dir,item+".nii.gz")) for item in available_mods]

        ##############################################
        # Do preprocessing in all modalities available

        for filename in [os.path.join(w_dir,item)+".nii.gz" for item in available_mods]:
            # Check if preprocessing has already been done and if so do not do it again
            if not os.path.exists(filename.replace(".nii.gz","_preproc.nii.gz")):
                preprocess(filename, filename.replace(".nii.gz","_preproc.nii.gz"))


        ##############################################
        # Find the modality that will be the "main" one from which to calculate
        # the transformation to standard space.
        # It will be the most isotropic according to the aspect ratios of the voxel sizes
        chosen_mod = None
        for mod in available_mods:
            # If isotropic, select right away
            if resolutions[available_mods.index(mod)]==[1, 1, 1]:
                chosen_mod = mod
                break # Break cycle because we found the best one

            if chosen_mod is None: # If we haven't selected one, select the current
                chosen_mod=mod

            # If we already selected one, but it is not isotropic, change to the current one if
            # the current one is more isotropic than the one we have
            else:
                res_chosen_mod = resolutions[available_mods.index(chosen_mod)]
                res_mod = resolutions[available_mods.index(mod)]
                # Compare isotropy and keep the modality that is more isotropic (closer to 1)
                if abs(1-calculate_isotropy(res_mod)) < abs(1-calculate_isotropy(res_chosen_mod)):
                    chosen_mod=mod

        if chosen_mod is None:
            continue # in that week no images were acquired (Note: this never happens)



        ##############################################
        # Register "main" image to standard space and save transformation matrix

        print("Registering " + chosen_mod + " to standard space")

        # Check if registration is done
        if not os.path.exists(os.path.join(w_dir,chosen_mod+"_tostd.nii.gz")):
            # Register to T1 atlas if the main modality is T1 with or without contrast
            if chosen_mod in ["CT1", "T1"]:
                register(static=os.path.join(ATLAS_FOLDER,"sri24_T1_3D.nii.gz"),
                            moving=os.path.join(w_dir,chosen_mod+"_preproc.nii.gz"),
                            output_image=os.path.join(w_dir,chosen_mod+"_tostd.nii.gz"),
                            transform_type="affine",
                            out_affine=os.path.join(w_dir,"matrix2std.txt"))
            else: # Register to T2 atlas
                register(static=os.path.join(ATLAS_FOLDER,"sri24_T2_3D.nii.gz"),
                            moving=os.path.join(w_dir,chosen_mod+"_preproc.nii.gz"),
                            output_image=os.path.join(w_dir,chosen_mod+"_tostd.nii.gz"),
                            transform_type="affine",
                            out_affine=os.path.join(w_dir,"matrix2std.txt"))


        # Check whether there are other modalities to transform besides the main one
        to_do=available_mods.copy()
        to_do.remove(chosen_mod)

        if len(to_do)==0:
            continue # there are no other images so we can go on to the next time point

        # Iterate through the rest of the images to be transformed
        print("Registering other images to the main image and then to standard space")
        for image in to_do:
            # Check if image was already registered
            if not os.path.exists(os.path.join(w_dir,image+"_tostd.nii.gz")):
                # Register to main image
                register(static=os.path.join(w_dir,chosen_mod+"_preproc.nii.gz"),
                        moving=os.path.join(w_dir,image+"_preproc.nii.gz"),
                        transform_type="rigid",
                        output_image=os.path.join(w_dir,image+"_to"+chosen_mod+".nii.gz"))

                # Apply transformation matrix from before to register to standard space
                if chosen_mod in ["CT1", "T1"]:
                    apply_transform(static=os.path.join(ATLAS_FOLDER,"sri24_T1_3D.nii.gz"),
                                    moving=os.path.join(w_dir,image+"_to"+chosen_mod+".nii.gz"),
                                    matrix=os.path.join(w_dir,"matrix2std.txt"),
                                    output_image=os.path.join(w_dir,image+"_tostd.nii.gz"))
                else:
                    apply_transform(static=os.path.join(ATLAS_FOLDER,"sri24_T2_3D.nii.gz"),
                                    moving=os.path.join(w_dir,image+"_to"+chosen_mod+".nii.gz"),
                                    matrix=os.path.join(w_dir,"matrix2std.txt"),
                                    output_image=os.path.join(w_dir,image+"_tostd.nii.gz"))

                # Delete unnecessary files
                force_delete_file(os.path.join(w_dir,image+"_to"+chosen_mod+".nii.gz"))
                force_delete_file(os.path.join(w_dir,image+"_preproc.nii.gz"))
        
        force_delete_file(os.path.join(w_dir,chosen_mod+"_preproc.nii.gz"))
        force_delete_file(os.path.join(w_dir,"matrix2std.txt"))

