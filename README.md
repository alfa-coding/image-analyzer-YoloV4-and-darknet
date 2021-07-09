# image-analyzer-functionalities

This respository provides the necessary tools to perform object detection using both the Darknet and TensorFlow API

## First create the environment:

`$conda env create -f analyzer-env.yml `

## To activate this environment, use:

`$ conda activate analyzer-env`

## To deactivate an active environment, use:

 `$ conda deactivate`

 ## get the tiny weigths

`$wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights`

Then, locate them within the assets folder.

# To run the test:

`$pytest`