# TensorFlow Image Segmentation for Brain Tumors
This tutorial offers a reference implementation on an Image Segmentation use case in detecting Brain Tumors based on MRI images. The framework used in this example is Tensorflow, implementing Unet model to train and evalutate on a set of brain scans.

## Key implementation details

There 2 stages in this example:
1. Data Preprocessing: In this stage, we will download the large BraTs dataset or use the existing small dataset which is in `.nii` format, process and covert `.nii` images to Keras compatible format `hdf5`.
2. Training Unet: In this stage, we will train the processed `hdf5` data using Unet topology with Intel-optimized Tensorflow along with  best run-time parameters to enable maximum performance on CPU.

The large Brats dataset is about 7GB and takes nearly 6+ hours to train the Unet model. So, for better user-experience for first-time users, we have shipped a small subset of the BraTs dataset called `small_dataset` which executes fairly faster. 

## Pre-requirement

TensorFlow is ready for use once you finish the Intel AI Analytics Toolkit installation, and have run post installation script.

You can refer to the oneAPI [main page](https://software.intel.com/en-us/oneapi) for toolkit installation, and the Toolkit [Getting Started Guide for Linux](https://software.intel.com/en-us/get-started-with-intel-oneapi-linux-get-started-with-the-intel-ai-analytics-toolkit) for post-installation steps and scripts.

## Activate conda environment With Root Access

Please follow the Getting Started Guide steps (above) to set up your oneAPI environment with the setvars.sh script. Then navigate in linux shell to your oneapi installation path, typically `<install_dir>/intel/inteloneapi`. Activate the conda environment with the following command:

#### Linux
```
source setvars.sh
source activate tensorflow
```

## Activate conda environment Without Root Access (Optional)

By default, the Intel AI Analytics toolkit is installed in the inteloneapi folder, which requires root privileges to manage it. If you would like to bypass using root access to manage your conda environment, then you can clone your desired conda environment using the following command:

#### Linux
```
source setvars.sh
conda create --name user_tensorflow --clone tensorflow
```

Then activate your conda environment with the following command:

```
source activate user_tensorflow
```

### Install Additional Dependencies

Install the following packages required to train unet model and processing neuro-imaging file formats. You can use the `conda install` command in your shell:

    conda install keras tqdm h5py psutil pydot graphviz nb_conda_kernels
    pip install nibabel

## License  


## How to Build and Run 
1. Go to the code example location<br>
2. Enter command `jupyter notebook` if you have GUI support <br>
or<br>
2a. Enter command `jupyter notebook --no-browser --port=8888` on a remote shell <br>
2b. Open the command prompt where you have GUI support, and forward the port from host to client<br>
2c. Enter `ssh -N -f -L localhost:8888:localhost:8888 <userid@hostname>`<br>
2d. Copy-paste the URL address from the host into your local browser to open the jupyter console<br>
3. Go to `Data_Preprocess.ipynb` and run each cell to process and covert the dataset<br>
4. Go to ` Train_Unet.ipynb` and run each cell to train the Unet model.

Note, In jupyter page, please choose the right 'kernel'. In this case, please choose it in menu 'Kernel' -> 'Change kernel' -> Python [conda env:intel_tf]

In case, if the AI toolkit is installed with root access, you may experience trouble installing new conda packages with user account such as EnviornmentNotWritableError. Please provide write permissions to the tensorflow environment by executing the below command

`sudo chmod 777 -R <ai kit install dir>/envs/tensorflow`
