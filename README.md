# First-Aid-CXR-AI
First-aid Chest-X ray Artificial Intelligence  which can support for several tasks: Gender, View, Vendor, Age and Abnormal Features.

## Prerequisites

- Python 3.7.+
- PyTorch 1.2.0+
- Python Packages: ./requirements

---
## Demo (w/ Docker)

### 1. Docker and Nvidia-docker installation
In order to load our docker image, you need to install `Docker` and `Nvidia-docker`.
- Docker : 18.03.0-ce ([installation](https://docs.docker.com/install/linux/docker-ce/ubuntu/#os-requirements))
- Nvidia-docker : v2.0.3 ([installation](https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0))))

Please follow the instructions described in the above links. Estimated installation time is about **10** minutes.

### 2. Build image
You have to follow up the below instruction to make docker image

- Clone this repo
- Download the pre-trained model for demo from Dropbox
: https://www.dropbox.com/s/9ml7dttcn8tfshf/models.zip?dl=0

```sh
# Dowload model zip file
$ cd ./models
$ wget -O models.zip https://www.dropbox.com/s/9ml7dttcn8tfshf/models.zip?dl=0
# unzip the zip file
$ unzip models.zip

# Remove file
$ rm models.zip
$ cd ..
```
- Make docker
```sh
$ sudo nvidia-docker build . -t cxr-ai-lmic:1.0 -f config_dir/Dockerfile
```

### 3. Run image
Open the terminal
```sh
$ export WORK_DIR="/path/to/work/dir"
$ export INPUT_DIR="$WORK_DIR/input_dir"
$ export OUTPUT_DIR="$WORK_DIR/output_dir"

# Run docker image (background option: -d)
$ sudo nvidia-docker run -d --gpus all -v $OUTPUT_DIR:/usr/app/output_dir -v $INPUT_DIR:/usr/app/input_dir --name cxr_ai -it cxr-ai-lmic:1.0 /bin/bash
```

### 4. Run demo
Now, you are ready to enjoy our `CXR AI`!

```sh
$ sudo nvidia-docker exec -i -t cxr_ai python demo_cxr.py
```

**[Input preparation before the model runs]**
- For dicom inputs: you have to copy dicom files as you run to `/input_dir/DICOM`
- For image inputs: you have to copy any image files as you run to `/input_dir/IMAGEFILE`

**[Expected outputs]**
- Prediction : Predicted values for every task saved to `/output_dir/Classification/prediction.txt`
- Probability: Predicted probabilities for every task saved to `/output_dir/Classification/probability.txt`
- Grad-CAM   : Grad-CAM for 6 abnormal features saved to `/output_dir/PDF/<inpu-file-name>_<abnormal-feature-name>_hmp.pdf`
