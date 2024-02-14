#!/bin/bash


# Step 1: Planning and Preprocessing for the target peds dataset
nnUNet_plan_and_preprocess -t 711  

# Step 2: Planning and Preprocessing for the source adult dataset
nnUNet_plan_and_preprocess -t 713 -overwrite_plans <PATH_TO_NNUNET_DIR>/nnunet/nnunet/RESULTS/nnUNet/3d_fullres/Task711_Ped_Fresh/nnUNetTrainerV3_100epochs__nnUNetPlansv2.1/plans.pkl -overwrite_plans_identifier nnUNetTrainerV3_100epochs__nnUNetPlansv2.1 -pl3d ExperimentPlanner3D_v21_Pretrained -tf 100

# Step 3 Pretraining on Adult data for 100 epochs
(trap 'kill 0' SIGINT; CUDA_VISIBLE_DEVICES=6 nnUNet_train 3d_fullres nnUNetTrainerV3_100epochs 713 0 --npz -p nnUNetPlans_pretrained_nnUNetTrainerV3_100epochs__nnUNetPlansv2.1 & CUDA_VISIBLE_DEVICES=4 nnUNet_train 3d_fullres nnUNetTrainerV3_100epochs 713 1 --npz -p nnUNetPlans_pretrained_nnUNetTrainerV3_100epochs__nnUNetPlansv2.1 & CUDA_VISIBLE_DEVICES=3 nnUNet_train 3d_fullres nnUNetTrainerV3_100epochs 713 2 --npz -p nnUNetPlans_pretrained_nnUNetTrainerV3_100epochs__nnUNetPlansv2.1 & CUDA_VISIBLE_DEVICES=2 nnUNet_train 3d_fullres nnUNetTrainerV3_100epochs 713 3 --npz -p nnUNetPlans_pretrained_nnUNetTrainerV3_100epochs__nnUNetPlansv2.1 & CUDA_VISIBLE_DEVICES=1 nnUNet_train 3d_fullres nnUNetTrainerV3_100epochs 713 4 --npz -p nnUNetPlans_pretrained_nnUNetTrainerV3_100epochs__nnUNetPlansv2.1 & wait)

# Step 4: To run a longer training schedule - 4000 epochs with no mirroring augemtation
(trap 'kill 0' SIGINT; CUDA_VISIBLE_DEVICES=7 nnUNet_train 3d_fullres nnUNetTrainerV5_4000epochs_nomirror 711 0 -pretrained_weights <PATH_TO_NNUNET_DIR>/nnunet<PATH_TO_NNUNET_DIR>/nnunet/RESULTS/nnUNet/3d_fullres/Task713_Peds_Fine/nnUNetTrainerV3_100epochs__nnUNetPlans_pretrained_nnUNetTrainerV3_100epochs__nnUNetPlansv2.1/fold_0/model_final_checkpoint.model & CUDA_VISIBLE_DEVICES=6 nnUNet_train 3d_fullres nnUNetTrainerV5_4000epochs_nomirror 711 1 -pretrained_weights <PATH_TO_NNUNET_DIR>/nnunet<PATH_TO_NNUNET_DIR>/nnunet/RESULTS/nnUNet/3d_fullres/Task713_Peds_Fine/nnUNetTrainerV3_100epochs__nnUNetPlans_pretrained_nnUNetTrainerV3_100epochs__nnUNetPlansv2.1/fold_1/model_final_checkpoint.model & CUDA_VISIBLE_DEVICES=5 nnUNet_train 3d_fullres nnUNetTrainerV5_4000epochs_nomirror 711 2 -pretrained_weights <PATH_TO_NNUNET_DIR>/nnunet<PATH_TO_NNUNET_DIR>/nnunet/RESULTS/nnUNet/3d_fullres/Task713_Peds_Fine/nnUNetTrainerV3_100epochs__nnUNetPlans_pretrained_nnUNetTrainerV3_100epochs__nnUNetPlansv2.1/fold_2/model_final_checkpoint.model & CUDA_VISIBLE_DEVICES=4 nnUNet_train 3d_fullres nnUNetTrainerV5_4000epochs_nomirror 711 3 -pretrained_weights <PATH_TO_NNUNET_DIR>/nnunet<PATH_TO_NNUNET_DIR>/nnunet/RESULTS/nnUNet/3d_fullres/Task713_Peds_Fine/nnUNetTrainerV3_100epochs__nnUNetPlans_pretrained_nnUNetTrainerV3_100epochs__nnUNetPlansv2.1/fold_3/model_final_checkpoint.model & CUDA_VISIBLE_DEVICES=3 nnUNet_train 3d_fullres nnUNetTrainerV5_4000epochs_nomirror 711 4 -pretrained_weights <PATH_TO_NNUNET_DIR>/nnunet<PATH_TO_NNUNET_DIR>/nnunet/RESULTS/nnUNet/3d_fullres/Task713_Peds_Fine/nnUNetTrainerV3_100epochs__nnUNetPlans_pretrained_nnUNetTrainerV3_100epochs__nnUNetPlansv2.1/fold_4/model_final_checkpoint.model & wait)

# Step 5: Generate predictions
CUDA_VISIBLE_DEVICES=6 nnUNet_predict -i <PATH_TO_NNUNET_DIR>/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task711_Ped_Fresh/imagesTs -o "$RESULTS_FOLDER/Results_711_finetuned_100nomirror" -t 711 -m 3d_fullres -tr nnUNetTrainerV5_100epochs_nomirror --num_threads_preprocessing 30 --num_threads_nifti_save 30 --disable_tta

echo "All commands have finished."
