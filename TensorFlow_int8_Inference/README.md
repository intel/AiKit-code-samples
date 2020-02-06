# TensorFlow int8 Inference
This code example provides a sample code to run ResNet50 low precision inference on Intel's pretrained int8 model

## Key implementation details
The example uses Intel's pretrained model published as part of Intel Model Zoo. The example also illustrates how to utilize TensorFlow and MKL run time settings to maximize CPU performance on ResNet50 workload

## Pre-requirement

TensorFlow is ready for use once you finish the Intel AI Analytics Toolkit installation, and have run post installation script.

You can refer to the oneAPI [main page](https://software.intel.com/en-us/oneapi) for toolkit installation, and the Toolkit [Getting Started Guide for Linux](https://software.intel.com/en-us/get-started-with-intel-oneapi-linux-get-started-with-the-intel-ai-analytics-toolkit) for post-installation steps and scripts.

## Activate conda environment

Please follow the Getting Started Guide steps (above) to set up your oneAPI environment with the setvars.sh script. Then navigate in linux shell to your oneapi installation path, typically `~/intel/inteloneapi`. Activate the conda environment with the following command:

#### Linux
```
source activate tensorflow
```

## Install Jupyter Notebook 
```
conda install jupyter nb_conda_kernels
```

## How to Build and Run 
1. Go to the code example location<br>
2. Enter command `jupyter notebook` if you have GUI support <br>
or<br>
2a. Enter command `jupyter notebook --no-browser --port=8888` on a remote shell <br>
2b. Open the command prompt where you have GUI support, and forward the port from host to client<br>
2c. Enter `ssh -N -f -L localhost:8888:localhost:8888 <userid@hostname>`<br>
2d. Copy-paste the URL address from the host into your local browser to open the jupyter console<br>
3. Go to `Tensorflow_resnet50_int8_inference.ipynb` and run each cell to create sythetic data and run int8 inference

Note, In jupyter page, please choose the right 'kernel'. In this case, please choose it in menu 'Kernel' -> 'Change kernel' -> Python [conda env:tensorflow]

## License  
This code sample is licensed under MIT license