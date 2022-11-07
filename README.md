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
mkdir data
python -m lib.utils.download_models
cd data/
unzip models.zip
ln -s <IMU_Mocap data path> ./IMU_Mocap_Output_Data
cd ../
```

<br><br>
The required data folder structure would be as below
```
$ Directory tree
.
├── data\
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
