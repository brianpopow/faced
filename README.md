# Facedetector: *faced*

This is a fork of [**faced**](https://github.com/iitzco/faced), a (near) real time face detection running on the CPU.

<p align="center">
  <img src="examples/demo_yolo.gif"/>
</p>


This fork converts the model from tensorflow 1.0 to ONNX and uses the .Net [ONNX runtime](https://github.com/microsoft/onnxruntime) to run the face detection with csharp. The image processing is done with [ImageSharp](https://github.com/SixLabors/ImageSharp).

Unfortunatly converting the face corrector model (see describtion below) to ONNX did not work out. The results do not match the original model.
But even without it the results look pretty ok and its reasonable fast. It takes **25 ms** on a Intel Core I7 6700K.

There is an example project in the subfolder `csharp.`

#### Benchmark Results

```
BenchmarkDotNet=v0.13.1, OS=Windows 10.0.19043.1645 (21H1/May2021Update)
Intel Core i7-6700K CPU 4.00GHz (Skylake), 1 CPU, 8 logical and 4 physical cores
.NET SDK=6.0.300-preview.22204.3
  [Host]     : .NET 5.0.16 (5.0.1622.16705), X64 RyuJIT
  DefaultJob : .NET 5.0.16 (5.0.1622.16705), X64 RyuJIT


|        Method |     Mean |    Error |   StdDev |
|-------------- |---------:|---------:|---------:|
| FaceDetection | 25.39 ms | 0.667 ms | 1.872 ms |

// * Hints *
Outliers
  FaceDetectionBenchmark.FaceDetection: Default -> 9 outliers were removed (31.78 ms..34.66 ms)

// * Legends *
  Mean   : Arithmetic mean of all measurements
  Error  : Half of 99.9% confidence interval
  StdDev : Standard deviation of all measurements
  1 ms   : 1 Millisecond (0.001 sec)
````


Example results:
<p align="center">
  <img src="examples/example.png"/>
</p>

#### ONNX conversion

The conversion was done as following:

```
python -m tf2onnx.convert --verbose --input face_yolo.pb --opset 15 --inputs img:0 --inputs-as-nchw img --use_default img --outputs prob:0,x_center:0,y_center:0,w:0,h:0 --output faced.onnx
```

## How to install

```bash
$ pip install git+https://github.com/iitzco/faced.git
```

> Soon to be available on `PyPI`.

## How to use

### As library

```python
import cv2

from faced import FaceDetector
from faced.utils import annotate_image

face_detector = FaceDetector()

img = cv2.imread(img_path)
rgb_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

# Receives RGB numpy image (HxWxC) and
# returns (x_center, y_center, width, height, prob) tuples.
bboxes = face_detector.predict(rgb_img, thresh)

# Use this utils function to annotate the image.
ann_img = annotate_image(img, bboxes)

# Show the image
cv2.imshow('image',ann_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### As command-line program

```bash
# Detection on image saving the output
$ faced --input imgs/demo.png --save
```

or

```bash
# Live webcam detection
$ faced --input webcam
```

or

```bash
# Detection on video with low decision threshold
$ faced --input imgs/demo.mp4 --threshold 0.5
```

See `faced --help` for more information.

## Examples

<p float="left">
  <img src="examples/arg-faced.png" width="400" />
  <img src="examples/gino-faced.png" width="400" />
</p>
<p float="left">
  <img src="examples/foo-faced.png" width="400" />
  <img src="examples/tri-faced.png" width="200" />
</p>
<p float="left">
  <img src="examples/phelps-faced.png" width="400" />
  <img src="examples/will-faced.png" width="400" />
</p>

## Performance

CPU (i5 2015 MBP)          |  GPU (Nvidia TitanXP)
:-------------------------:|:-------------------------:
~5 FPS  | > 70 FPS

## Comparison with Haar Cascades

Haar Cascades are one of the most used face detections models. Here's a comparison with OpenCV's implementation showing *faced* robustness.

*faced*             |  Haar Cascade
:-------------------------:|:-------------------------:
![](examples/demo_yolo.gif)  |  ![](examples/demo_haar.gif)
![](examples/foo-faced.png)  |  ![](examples/foo-haar.png)
![](examples/gino-faced.png)  |  ![](examples/gino-haar.png)

## About *faced*

*faced* is an ensemble of 2 deep neural networks (implemented using **tensorflow**) designed to run at Real Time speed in CPUs.

#### Stage 1:

A custom fully convolutional neural network (FCNN) implementation based on [YOLO](https://pjreddie.com/darknet/yolo/). Takes a 288x288 RGB image and outputs a 9x9 grid where each cell can predict bounding boxes and probability of one face.

<p align="center">
  <img src="examples/cnn-scheme.png"/>
</p>


#### Stage 2:

A custom standard CNN (Convolutions + Fully Connected layers) is used to take a face-containing rectangle and predict the face bounding box. This is a fine-tunning step. (outputs of Stage 1 model is not so accurate by itself, this is a *corrector* step that takes the each bouding box predicted from the previous step to improve bounding box quality.)

<p align="center">
  <img src="examples/face-corrector-scheme.png"/>
</p>


### Why not just perform transfer learning on trained YOLO (or MobileNet+SSD) ?

Those models were designed to support multiclass detection (~80 classes). Because of this, these networks have to be powerfull enough to capture many different low and high level features that allow them to understand the patterns of very different classes. Powerful in this context means large amount of learnable parameters and hence big networks. These big networks cannot achieve real time performance on CPUs. [1]

This is an overkill for the simple task of just detecting faces. This work is a proof of concept that lighter networks can be designed to perform simpler tasks that do not require relatively large number of features.

[1] Those models cannot perform Real Time on CPU (YOLO at least). Even tiny-yolo version cannot achieve 1 fps on CPU (tested on 2015 MacBook Pro with 2.6 GHz Intel Core i5).

### How was it trained?

Training was done with [WIDER FACE](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) dataset on Nvidia Titan XP GPU.

> If you are interested in the training process and/or data preprocessing, just raise an `issue` and we'll discuss it there.

### How to run on GPU?

Just install `tensorflow-gpu` instead of `tensorflow`.

### Status

🚧 Work in progress 🚧
Models will be improved and uploaded.

**This is not a Production ready system. Use it at your own risk.**

