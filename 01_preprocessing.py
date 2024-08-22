#%%
# Imports
print("Importing")
import os

from utils import *

#%% 
# Extract first volume of SRI atlases (T1 and T2)
extract_firstvolume("/home/amatoso/phd/atlases/SRI24_T1_brain.nii",
                    "/home/amatoso/phd/atlases/sri24_T1_3D.nii.gz")

extract_firstvolume("/home/amatoso/phd/atlases/SRI24_T2_brain.nii",
                    "/home/amatoso/phd/atlases/sri24_T2_3D.nii.gz")

# Define priority of modalities
mods=["CT1","T1","T2","FLAIR"]

data_dir="/home/amatoso/phd/lumiere/data"

for patient in [p for p in sorted(os.listdir(data_dir))]:
    p_dir = os.path.join(data_dir,patient)
    for week in [p for p in sorted(os.listdir(p_dir))]:
        print("Doing "+patient+", "+week)
        w_dir = os.path.join(p_dir,week)

        # List the modalities the timepoint has and their resolutions
        available_mods = [item for item in mods if item+".nii.gz" in os.listdir(w_dir)]
        resolutions= [get_resolution(os.path.join(w_dir,item+".nii.gz")) for item in mods if item+".nii.gz" in os.listdir(w_dir)]

        ##############################################
        # Do preprocessing in all modalities available
        for filename in [os.path.join(w_dir,item)+".nii.gz" for item in available_mods]:
            if not os.path.exists(filename.replace(".nii.gz","_preproc.nii.gz")):
                preprocess(filename, filename.replace(".nii.gz","_preproc.nii.gz"))

        ##############################################
        # Find the modality that will be the "main" one from which to calculate the transformation to standard space. It will be the most isotropic according to the aspect ratios of the voxel sizes
        chosen_mod = None
        for mod in available_mods:
            if (resolutions[available_mods.index(mod)]==[1,1,1]).all(): # If isotropic, select right away
                chosen_mod = mod
                break # Break cycle because we found the best one

            elif chosen_mod is None: # If we haven't selected one, select the current
                    chosen_mod=mod

            else: # If we already selected one, but it is not isotropic, change to the current one if the current one is more isotropic than the one we have
                res_chosen_mod=resolutions[available_mods.index(chosen_mod)]
                res_mod=resolutions[available_mods.index(mod)]
                if abs(1-calculate_isotropy(res_mod)) < abs(1-calculate_isotropy(res_chosen_mod)): # Compare isotropy
                    chosen_mod=mod

        if chosen_mod is None: continue # in that week no images were acquired (Note: this never happens)

 
    
        ##############################################
        # Register "main" image to standard space and save transformation matrix

        print("Registering " + chosen_mod + " to standard space")   
        if chosen_mod in ["CT1", "T1"]:
            register(static="/home/amatoso/phd/atlases/sri24_T1_3D.nii.gz", 
                           moving=os.path.join(w_dir,chosen_mod+"_preproc.nii.gz"), 
                           output_image=os.path.join(w_dir,chosen_mod+"_tostd.nii.gz"),
                           type="affine", 
                           out_affine=os.path.join(w_dir,"matrix2std.txt"))
        else:
            register(static="/home/amatoso/phd/atlases/sri24_T2_3D.nii.gz", 
                           moving=os.path.join(w_dir,chosen_mod+"_preproc.nii.gz"), 
                           output_image=os.path.join(w_dir,chosen_mod+"_tostd.nii.gz"),
                           type="affine",
                           out_affine=os.path.join(w_dir,"matrix2std.txt"))
            

        # Check whether there are other modalities to transform besides the main one
        to_do=available_mods.copy()
        to_do.remove(chosen_mod)

        if len(to_do)==0: continue # there are no other images so we can go on to the next timepoint
            
        # Iterate through the rest of the images to be transformed
        print("Registering other images to the main image and then to standard space")
        for image in to_do:
            # Register to main image
            register(static=os.path.join(w_dir,chosen_mod+"_preproc.nii.gz"), 
                     moving=os.path.join(w_dir,image+"_preproc.nii.gz"), 
                     type="rigid", 
                     output_image=os.path.join(w_dir,image+"_to"+chosen_mod+".nii.gz"))
            
            # Apply transformation matrix from before to register to standard space
            if chosen_mod in ["CT1", "T1"]:
                apply_transform(static="/home/amatoso/phd/atlases/sri24_T1_3D.nii.gz",
                                moving=os.path.join(w_dir,image+"_to"+chosen_mod+".nii.gz"),
                                matrix=os.path.join(w_dir,"matrix2std.txt"),
                                output_image=os.path.join(w_dir,image+"_tostd.nii.gz"))
            else:
                apply_transform(static="/home/amatoso/phd/atlases/sri24_T2_3D.nii.gz",
                                moving=os.path.join(w_dir,image+"_to"+chosen_mod+".nii.gz"),
                                matrix=os.path.join(w_dir,"matrix2std.txt"),
                                output_image=os.path.join(w_dir,image+"_tostd.nii.gz"))
            
            # Delete unnecessary files
            force_delete_file(os.path.join(w_dir,image+"_to"+chosen_mod+".nii.gz"))
            force_delete_file(os.path.join(w_dir,chosen_mod+"_tostandard.nii.gz"))
            force_delete_file(os.path.join(w_dir,image+"_tostandard.nii.gz"))
            


        


#%%


""" 
# Plot images
mods=["T1","T2","FLAIR"]
for mod in mods:
    plot_slices(["/home/amatoso/phd/atlas_t1.nii.gz","./registration_84/"+mod+"_to_standard.nii.gz"])
    plt.suptitle(mod)

# Check sizes and resolutions
dir="/home/amatoso/phd/registration"
for file in os.listdir(dir):
    sub.run(["mrinfo", dir+"/"+file])
"""

    # %%
