[![License](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause) 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/combining-hough-transform-and-deep-learning/ecg-digitization-on-physionet-challenge-2024)](https://paperswithcode.com/sota/ecg-digitization-on-physionet-challenge-2024?p=combining-hough-transform-and-deep-learning)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

<h1 align='center'> ECG Digitiser – PhysioNet Challenge Winner (2024)<br> [<a href="https://arxiv.org/abs/2410.14185">arXiv Paper</a>] </h1>


**ECG Digitiser** is the state-of-the-art solution for converting ECG printouts into digital signals, enabling effective data extraction from legacy medical records. Our method combines the Hough Transform with deep learning, and achieved 1st place in the George B. Moody PhysioNet Challenge 2024.

This repository contains an open-source implementation, including:

- ✅ **Digitisation scripts** with straightforward usage examples.
- ✅ **Training scripts** for training [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) to segment ECG printouts.
- ✅ **Pretrained ECG segmentation models** for immediate use.
- ✅ **Synthetic ECG Printout Generation** using [ecg-image-kit](https://github.com/alphanumericslab/ecg-image-kit) and the [PTB-XL dataset](https://www.nature.com/articles/s41597-020-0495-6).


## 🚀 Quick Start

### Install
```bash
https://github.com/felixkrones/ECG-Digitiser.git
cd ECG-Digitiser
git lfs install # If not already installed
git lfs pull # Only to also pull the weights
conda create -n ecgdig python=3.11 # Or any other package manager
conda activate ecgdig
pip install -r requirements.txt
```

At the moment, the official [nnU-Net](https://github.com/felixkrones/nnUNet.git) repository contains a bug and is not working with RGB png images. Please use the following for now:
```bash
cd nnUNet
pip install .
cd ..
```

### Use
```bash
python -m src.run.digitize -d data_folder -o output_folder
```

Where `data_folder` is the folder containing the ECG images to be digitized and `output_folder` is the folder where the outputs will be saved.


## 🔍 Example Results

### ECG Segmentation
<p align="center">
    <img src="./figures/segmentation.png" alt="ECG Segmentation Example" width="800"/>
    <br><em>Example of ECG segmentation from a printed ECG image.</em>
</p>

### ECG Vectorisation
<p align="center">
    <img src="./figures/vectorisation.png" alt="ECG Vectorisation Example" width="800"/>
    <br><em>Example of vectorised ECG signal reconstructed from segmentation.</em>
</p>


## 🗂️ Repository Structure

- `ecg-image-generator`: Generate synthetic ECG images using [ecg-image-kit](https://github.com/alphanumericslab/ecg-image-kit) and the [PTB-XL dataset](https://www.nature.com/articles/s41597-020-0495-6).
- `figures`: Example images for the README.
- `models`: Trained model weights.
- `nnUNet`: Segmentation model [nnU-Net](https://github.com/MIC-DKFZ/nnUNet).
- `config.py`: Main configuration file.
- `src/run/digitize.py`: Digitisation pipeline script.
- `src/ptb_xl`: Prepare [PTB-XL dataset](https://www.nature.com/articles/s41597-020-0495-6).
- `src/utils`: Helper functions.


## 📚 Data

To run the code, you need different kind of data.
If you are using the PTB-XL dataset, see below under `Using the PTB-XL dataset` on how to prepare the data.

- **Training data for the segmentation model:**
In order to train the segmantation model, you need to have the data in the format as described in `nnUNet/documentation/dataset_format.md`. (For the following description, we assume the folder containing the data is called `Dataset500_Signals`.)

- **Data to run the digitization:**
A folder containing the ECG images to be digitized. Those images should match the ones used for training the segmentation model.
If you are using our pre-trained weights, those should match the images from the [2024 Challenge](https://physionetchallenges.org/2024/).


### Using the PTB-XL dataset

1. Download (and unzip) the [PTB-XL dataset](https://physionet.org/content/ptb-xl/) and [PTB-XL+ dataset](https://physionet.org/content/ptb-xl-plus/).
Replace the name of the folder (probably `ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3`) that contains a file named `ptbxl_database.csv` with `ptb-xl/`. 
Replace the name of the folder (probably `ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1`) that contains a folder called `labels` with `ptb-xl-p/`. 
Replace the paths below with the actual paths to the folders.

2. Add information from various spreadsheets from the PTB-XL dataset to the WFDB header files:

        python -m src.ptb_xl.prepare_ptbxl_data \
            -i ptb-xl/records500 \
            -pd ptb-xl/ptbxl_database.csv \
            -pm ptb-xl/scp_statements.csv \
            -sd ptb-xl-p/labels/12sl_statements.csv \
            -sm ptb-xl-p/labels/mapping/12slv23ToSNOMED.csv \
            -o ptb-xl/records500_prepared

3. [Generate synthetic ECG images](https://github.com/alphanumericslab/ecg-image-kit/tree/main/codes/ecg-image-generator) on the dataset:

    1. Move into ecg-image-generator: `cd ecg-image-generator`
    2. Deactivate the current env (`conda deactivate`) and create a new one by following the instructions in the README of the generator repo (`ecg-image-generator/README.md`) (this means running `conda env create -f environment_droplet.yml` and then `conda activate ecg_gen`).
    3. Now you can run the following code. Careful though, this can take very long, around 10 min per subfolder (approx. 1000 files) or 4h in total and will increase the necessary disk space by approx. 15x, adding another 8GB for the 500Hz data. To test, better to run it on one single subfolder (e.g., add /00000):

            python gen_ecg_images_from_data_batch.py \
                -i ptb-xl/records500_prepared \
                -o ptb-xl/records500_prepared_w_images \
                --print_header \
                --store_config 2 \
                --mask_unplotted_samples

        For example:

            python gen_ecg_images_from_data_batch.py \
                -i ptb-xl/records500_prepared/00000 \
                -o ptb-xl/records500_prepared_w_images/00000 \
                -se 10 \
                --mask_unplotted_samples \
                --print_header \
                --store_config 2 \
                --lead_name_bbox \
                --lead_bbox \
                --random_print_header 0.7 \
                --calibration_pulse 0.6 \
                --fully_random \
                -rot 5 \
                --num_images_per_ecg 4 \
                --random_bw 0.1 \
                --run_in_parallel \
                --num_workers 8
                
    4. Deactivate the environment again (`conda deactivate`), move back to your original repo (`cd ..`) and activate the environment from above again (`conda activate env-name`).

4. Add the file locations and other information for the synthetic ECG images to the WFDB header files:

        python -m src.ptb_xl.prepare_image_data \
            -i ptb-xl/records500_prepared_w_images \
            -o ptb-xl/records500_prepared_w_images

5. Create more dense pixels for the masks:

        python -m src.ptb_xl.replot_pixels \
            --resample_factor 3 \
            --dir ptb-xl/records500_prepared_w_images \
            --run_on_subdirs \
            --num_workers 12

6. Convert it into the nnUNet format:

    We use the suggested splits from `ptbxl_database.csv`:

        python -m src.ptb_xl.create_train_test \
            -i ptb-xl/records500_prepared_w_images \
            -d ptb-xl/ptbxl_database.csv \
            -o ptb-xl/Dataset500_Signals

    For example:

        python -m src.ptb_xl.create_train_test \
            -i ptb-xl/records500_prepared_w_images \
            -d ptb-xl/ptbxl_database.csv \
            -o ptb-xl/Dataset500_Signals \
            --rgba_to_rgb \
            --gray_to_rgb \
            --mask \
            --mask_multilabel \
            --rotate_image \
            --plotted_pixels_key dense_plotted_pixels \
            --num_workers 8


## ▶️ Run

You can either use our model weights for the segmentation model or train your own model following the steps under `1.` below.

### 1. Train the segmentation model

First, one needs to train the segmentation model.
We use [nnU-Net](https://github.com/felixkrones/nnUNet.git); this involves multiple steps:

1. Set the environment variables for nnU-Net (just post the following in the terminal from where you want to run the code):

        # Set environment variables
        export nnUNet_raw='path to the folder that contains the Dataset500_Signals folder'
        export nnUNet_preprocessed='any folder path to save the preprocessed data'
        export nnUNet_results='any folder path to save the results'

        # Check
        echo ${nnUNet_raw}

2. Experiment planning and preprocessing

        nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity

    For example:

        nnUNetv2_plan_and_preprocess -d 500 --clean -c 2d --verify_dataset_integrity

3. Model training

        nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD

    For example:

        nnUNetv2_train 500 2d 0 -device cuda --c

    Or select the device (one per fold):

        CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 500 2d 1 --c

4. Determine the best configuration

        nnUNetv2_find_best_configuration DATASET_NAME_OR_ID -c CONFIGURATIONS

    For example:

        nnUNetv2_find_best_configuration 500 -c 2d --disable_ensembling


### 2. Run the digitization

1. Set the parameters in `config.py`.

2. Run the digitization pipeline by running:

        python -m src.run.digitize -d data_path -m model_path -o output_path -v

where

- `data_path` (input; required) is the folder containing the images to be digitized;
- `model_path` (input; required) is the folder containing the segmentation model folder `nnUNet_results`, e.g., `models/M3`;
- `output_path` is the folder where the outputs will be saved.


## 📄 Citation and Acknowledgements

If you use this code in your research, please cite our [paper](https://arxiv.org/abs/2410.14185):

    @article{krones2024combining,
        title={Combining Hough Transform and Deep Learning Approaches to Reconstruct ECG Signals From Printouts},
        author={Krones, Felix and Walker, Ben and Lyons, Terry and Mahdi, Adam},
        journal={arXiv:2410.14185},
        year={2024}
    }

    Krones F, Walker B, Lyons T, Mahdi A. Combining Hough Transform and Deep Learning Approaches to Reconstruct ECG Signals From Printouts. arXiv:2410.14185. 2024 Oct 18.

Additionally, please consider citing the [PhysioNet Challenge 2024](https://physionetchallenges.org/2024/), [ecg-image-kit](https://github.com/alphanumericslab/ecg-image-kit), [PTB-XL dataset](https://www.nature.com/articles/s41597-020-0495-6), and [nnU-Net](https://github.com/MIC-DKFZ/nnUNet).
