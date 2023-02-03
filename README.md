# JointAnglePrediction_Validate

## 1. Installation
Currently, the code repository has been validated on Pyhon 3.9, PyTorch 1.13.0 (CUDA 11.7). <br>
We recommend you to create a virtual environment and install all the dependencies. <br>
The following is the example commandlines for Anaconda users.

```
git clone <>
conda create -n jointangle-validation
conda activate jointangle-validation
pip install -r requirements.txt
```
<br><br>

## 2. Data structure
Run the following commandline (for Linux user)
```
mkdir dataset
python -m lib.utils.download_models
cd dataset/
unzip models.zip
ln -s <IMU_Mocap data path> ./IMU_Mocap_Output_Data
cd ../
```

<br><br>
The required data folder structure would be as below
```
$ Directory tree
.
├── dataset\
    ├── models\
    │    ├── checkpoints\ 
    │         ├── Running\
    │         │   ├── Ankle_model_kwargs.pkl
    │         │   ├── Ankle_model.pt
    │         │   ├── ...
    │         │
    │         └── Walking\
    │             ├── ...
    │
    └── IMU_Mocap_Output_Data\
        ├── HS001_Output_Data\
        ├── HS002_Output_Data\
        ├── HS003_Output_Data\
        ├── ...
```
<br><br>

## 3. Preprocess data
Run the following commandline to preprocess train & test data. 
``` 
python -m lib.data.preprocess_data
```
You will have `test_data_label.pt` and `train_data_label.pt` under the `IMU_Mocap_Output_Data` folder.

<br><br>

## 4. To train model, run finetune.py -be sure to set a configuration (the baseline is saved as baseline.yaml), as well as change any other training paraemters. See options in configs.


5. To run model, run_model.py
