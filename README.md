# Better-PediatricCT-segmentation

This repository contains the corresponding code for the paper "Children Are Not Small Adults: Addressing Limited Generalizability of an Adult Deep Learning Organ Segmentation Model to the Pediatric Population". 

## Dependencies
To reproduce all the experiments successfully, please clone this repository locally and initiate a conda virtual environment using the given `environment.yml` file using 

```
conda env create -f environment.yml
```

Depending on your GPU and CUDA version, please update Pytorch to ensure that you have the correct CUDA version of Pytorch installed on your system.

This code depends on nnUNet v1 for training and predicting all models. Please ensure that after setting up nnUNet, you have taken care to set up the required paths as shown [here](https://github.com/MIC-DKFZ/nnUNet/blob/nnunetv1/documentation/setting_up_paths.md).

## Datasets

This study employed the [TCIA Pediatric-CT-SEG](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=89096588) and the [Abdominal Multi-Organ Segmentation Chellenge (AMOS)](https://amos22.grand-challenge.org/) datasets. 

## Files

### [run_predictions_parallelized.sh](https://github.com/UM2ii/Better-PediatricCT-segmentation/blob/main/run_predictions_parallelized.sh):

1. **Paths and Base Command:**
   - Define input and output directories and the base command for nnUNet prediction.

2. **Loop Through GPU IDs:**
   - Loop through GPU IDs from 3 to 7.
   - Construct commands for each GPU, specifying the part ID and the number of parts.
   - Execute commands in the background.

3. **Wait for Completion:**
   - Wait for all background processes to finish.

4. **Completion Message:**
   - Print a completion message.

### [run_trainingpipeline_peds_nnunet.sh](https://github.com/UM2ii/Better-PediatricCT-segmentation/blob/main/run_trainingpipeline_peds_nnunet.sh):

1. **Welcome Message:**
   - Display a welcome message explaining the script's purpose.

2. **User Input:**
   - Prompt the user to input the task name identifier and the path to the testing data folder.

3. **Training:**
   - Initiate training on multiple GPUs for the specified task.
   - Wait for training processes to finish.

4. **Prediction:**
   - Create a directory for storing prediction results.
   - Perform prediction using trained model on the testing data.

### Additional Notes:
- Both scripts utilize nnUNet for training and prediction.
- They leverage multiple GPUs for parallel processing to expedite computation.
- The second script involves user interaction for task customization.

### [run_finetuning.sh](https://github.com/UM2ii/Better-PediatricCT-segmentation/blob/main/run_finetuning.sh):

#### Step 1: Planning and Preprocessing for the target peds dataset
Command: nnUNet_plan_and_preprocess -t 711
Description: This step involves planning and preprocessing the target pediatric dataset (Our dataset ID was set to 711) for training with nnU-Net. Planning includes generating configuration files and preprocessing involves data augmentation, normalization, and resizing to prepare the dataset for training.

#### Step 2: Planning and Preprocessing for the source adult dataset
Command: 
```
nnUNet_plan_and_preprocess -t 713 -overwrite_plans /home/akanhere/nnunet/home/akanhere/nnunet/RESULTS/nnUNet/3d_fullres/Task711_Ped_Fresh/nnUNetTrainerV3_100epochs__nnUNetPlansv2.1/plans.pkl -overwrite_plans_identifier nnUNetTrainerV3_100epochs__nnUNetPlansv2.1 -pl3d ExperimentPlanner3D_v21_Pretrained -tf 100
```
Description: Similar to Step 1, this step plans and preprocesses the source adult dataset (ID 713). The overwrite_plans and overwrite_plans_identifier flags specify that existing plans should be overwritten with new plans, and the pl3d flag indicates the type of planner to use (ExperimentPlanner3D_v21_Pretrained). -tf is the number of processes used for preprocessing the full-resolution data of the 2D U-Net and 3D U-Net.

#### Step 3: Pretraining on Adult data for 100 epochs
Command: This step involves training the model on the adult dataset (ID 713) for 100 epochs. The training is distributed across multiple GPUs, with each GPU handling training for a different fold of the data.
Description:
CUDA_VISIBLE_DEVICES=<gpu_id> nnUNet_train ...: This command initiates training on the specified GPU.
3d_fullres: Indicates the resolution at which the training is performed.
nnUNetTrainerV3_100epochs: Specifies the [custom trainer configuration]() and the number of epochs for training.
713 <fold_number>: Specifies the dataset and fold number for training.
--npz -p nnUNetPlans_pretrained_nnUNetTrainerV3_100epochs__nnUNetPlansv2.1: Specifies file formats for input and output and the path for the plans file.

#### Step 4: To run a longer training schedule - 4000 epochs with no mirroring augmentation
Command: Similar to Step 3, this step involves training the model on the target pediatric dataset (711) for 4000 epochs with no mirroring augmentation.
Description:
CUDA_VISIBLE_DEVICES=<gpu_id> nnUNet_train ...: Initiates training on the specified GPU.
3d_fullres: Indicates the resolution at which the training is performed.
nnUNetTrainerV5_4000epochs_nomirror: Specifies the [custom trainer configuration]() and the number of epochs for training (4000 epochs with no mirroring augmentation).
711 <fold_number>: Specifies the dataset and fold number for training.
-pretrained_weights <path_to_pretrained_model>: Specifies the path to the pre-trained model weights from Step 3.
#### Step 5: Generate predictions
Command:
```
CUDA_VISIBLE_DEVICES=6 nnUNet_predict -i /home/akanhere/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task711_Ped_Fresh/imagesTs \
-o "$RESULTS_FOLDER/Results_711_finetuned_100nomirror" -t 711 -m 3d_fullres -tr nnUNetTrainerV5_100epochs_nomirror \
--num_threads_preprocessing 30 --num_threads_nifti_save 30 --disable_tta
```
Description: This step generates predictions using the trained model. It specifies the input directory (-i), output directory (-o), target dataset (-t), model type (-m), trainer type (-tr), and other parameters for preprocessing and saving the predictions. Please note that since we disabled the mirroring augmentations, we need to specify the --disable_tta flag for nnunet to generate predictions correctly. 

### Additional Notes:
- The nnU-Net architecture used is a modified 3D U-Net, tailored for automated medical image segmentation.
- Modifications include removing additional mirroring augmentation and increasing training epochs to 4000 to mimic the TotalSegmentator pipeline.
- The combined pediatric and adult CT dataset was split into training, validation, and testing sets. (Refer to the paper for details)
- During inference, test-time augmentation with mirroring was disabled to ensure consistency.
- Rigorous validation assessed the model's generalization ability to unseen cases from both populations.
