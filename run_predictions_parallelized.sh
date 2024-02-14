#!/bin/bash

# Define paths to input and output directories
input_dir="<PATH_TO_NNUNET_DIR>/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task713_Peds_Fine/imagesTs"
output_dir="<PATH_TO_NNUNET_DIR>/nnunet/nnunet/RESULTS/Results_711_onAMOS713"

# Define the base command for nnUNet_predict
base_command="nnUNet_predict -i ${input_dir} -o ${output_dir} -t 711 -m 3d_fullres -tr nnUNetTrainerV5_4000epochs_nomirror --num_threads_preprocessing 30 --num_threads_nifti_save 30 --disable_tta"

# Loop through GPU IDs from 3 to 7
for gpu_id in {3..7}; do
    part_id=$((gpu_id - 3)) # Calculate part_id based on GPU ID
    num_parts=5              # Total number of parts

    # Construct the full command
    full_command="${base_command} --part_id=${part_id} --num_parts=${num_parts}"

    # Export CUDA_VISIBLE_DEVICES and run the command with the specified GPU ID
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    ${full_command} &
done

# Wait for all background processes to finish
wait

echo "All commands have finished."
