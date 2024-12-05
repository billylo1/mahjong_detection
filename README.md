# Mahjong Detection with YOLOv11
This repository contains a project for building a YOLOv11 model to detect different tiles in Mahjong. The project includes data preparation, model training, and inference.

## Installation

To get started, install the required packages by running:

```bash
pip install -r requirements.txt
```

## Training the Model

To train the YOLOv11 model, run the `training.ipynb` notebook. The training data is sourced from Roboflow and supplemented with an additional dataset from the repo jaheel/MJOD-2136. To download the training dataset from roboflow, setup `.env` file with your roboflow API key. 

## Data Preparation

The dataset from jaheel/MJOD-2136 is not in YOLO format. To convert it, use the provided script `convert_dataset.py`. This script will reformat the dataset into the YOLO format required for training. The sample code is provided in the `training.ipynb`.

## Manual Annotation Verification

Before training, it's important to verify the annotations to ensure data quality. You can manually check the annotations by cropping them using the `crop_annotations.py` script. 

To run the script, use the following command:

```bash
python crop_annotations.py --in_path <path/to/dataset>
```

## Inference

After training the model, you can use `predict.py` to perform inference and visualize the results on images.

To run the inference, use:

```bash
python predict.py -i sample.jpg
```

## Finding a winning hand
RANSAC is used to find the most likely winning hand. As the minimum number of tiles for winning is 14, and assuming that the image contains at most 10 non winning hand, we can calculate the number of iterations needed for the RANSAC algorithm, i.e.,
$k = \frac{log(1-p)}{log(1-w^n)}$ where $p=0.99, w=14/24, n=4$. For images containing too many detections, it is simply assumed that the winning hand is located in the lower half of the image to minimize the number of outliers.

To run the inference with enablign the locate winning hand feature, run:
```bash
python predict.py -i sample.jpg --find_winning_hand
```

## Results
![image](https://github.com/user-attachments/assets/29a2820b-682d-4124-b7b7-5a18c4c0779d)
![Screenshot from 2024-12-03 11-36-49](https://github.com/user-attachments/assets/5addf5f9-607a-4740-8aa3-2b6675907f70)
![Screenshot from 2024-12-03 11-52-10](https://github.com/user-attachments/assets/1f8f2d8d-2b3a-4f88-bcd9-7675fab418ab)
