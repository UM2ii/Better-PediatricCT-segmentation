#!/bin/bash
echo ==========================================
echo Welcome to the UM2ii nnUNet autorun script.
echo This script will automatically run the nnUNet training and prediction pipeline for a custom dataset.
echo ==========================================

echo Enter the task name identifier
read task_name

echo Path to the folder containing the testing data
read test_path

(trap 'kill 0' SIGINT; 
CUDA_VISIBLE_DEVICES=7 nnUNet_train 3d_fullres nnUNetTrainerV5_4000epochs_nomirror $task_name 0 --npz & 
CUDA_VISIBLE_DEVICES=6 nnUNet_train 3d_fullres nnUNetTrainerV5_4000epochs_nomirror $task_name 1 --npz & 
CUDA_VISIBLE_DEVICES=5 nnUNet_train 3d_fullres nnUNetTrainerV5_4000epochs_nomirror $task_name 2 --npz & 
CUDA_VISIBLE_DEVICES=4 nnUNet_train 3d_fullres nnUNetTrainerV5_4000epochs_nomirror $task_name 3 --npz & 
CUDA_VISIBLE_DEVICES=3 nnUNet_train 3d_fullres nnUNetTrainerV5_4000epochs_nomirror $task_name 4 --npz & 
wait) 

mkdir "$RESULTS_FOLDER/Results_Ped4000_$task_name"
nnUNet_predict -i "$test_path" -o "$RESULTS_FOLDER/Results_Ped4000_$task_name" -t "$task_name" -m 3d_fullres -tr nnUNetTrainerV5_4000epochs_nomirror

