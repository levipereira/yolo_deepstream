# YoloV7 Quantization Aware Training (with Enhancements)



## Changes

Changes
- The parameter `--size` has been replaced with `--img-size`. ( just to standardize with the original repository).
- `--img-size ` Specifies the image size for both training and testing.

### New Feature
- Added `scripts/generate_trt_engine.sh` script to generate TRT engine from ONNX
- Improved `scripts/eval-trt.py` to use pycocotools on Custom Dataset

#### Added Evaluation Report using PyCocoTools for Custom Dataset
New Feature Added:
- `--use-pycocotools`: Generates COCO annotation/evaluation format JSON for the custom dataset.

With this flag, users can utilize the pycocotools functionality. However, it requires the usage of the YOLOv7 repository from https://github.com/levipereira/yolov7, as it is the only supported repository to enable this feature.


#### Improved Input Dataset Loader
This change supports both COCO dataset and custom datasets, similar to the original YOLOv7 repository.

Parameter Added/Changed:
- The parameter `--cocodir` has been replaced with `--data`.
- `--data`: Specifies the path to the data.yaml file. (default: 'data/coco.yaml')
- `--hyp`: Specifies the path to the hyperparameters file. (default: 'data/hyp.scratch.p5.yaml')


#### Added Model Save Dir
New Parameters Added to `scripts/qat.py export/quantize/sensitive `:
- `--experiment`: Specifies the root directory for saving experiments. (default: `'experiments/<export/qat/sensitive>'`)
- `--project_name`: Specifies the project name for organizing experiments. (default: `current date in YYYYMMDDHH24MISS format`)


#### End-to-End support for ONNX export (Efficient NMS plugin).
New Parameters. Added to `scripts/qat.py export`:
- `--end2end`: Enables end-to-end ONNX export.
- `--topk-all`: Specifies the top-k objects for every image.
- `--simplify`: Facilitates simplifying the ONNX model.
- `--iou-thres`: Sets the IoU threshold for NMS.
- `--conf-thres`: Sets the confidence threshold for NMS.<br>
<br>Note that when using the end2end option with the `Detect() layer` (grid option) is automatically included.

#
## Description
 We use [TensorRT's pytorch quntization tool](https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization) to finetune training QAT yolov7 from the pre-trained weight, then export the model to onnx and deploy it with TensorRT. The accuray and performance can be found in below table.

|  Method   | Calibration method  | mAP<sup>val<br>0.5|mAP<sup>val<br>0.5:0.95 |batch-1 fps<br>Jetson Orin-X  |batch-16 fps<br>Jetson Orin-X  |weight|
|  ----  | ----  |----  |----  |----|----|-|
| pytorch FP16 | -             | 0.6972 | 0.5120 |-|-|[yolov7.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt)|
| pytorch PTQ-INT8  | Histogram(MSE)  | 0.6957 | 0.5100 |-|-|[yolov7_ptq.pt](https://drive.google.com/file/d/1AMymKjKMDmhuNSI3jzL6dv_Pc3rdDDj1/view?usp=sharing) [yolov7_ptq_640.onnx](https://drive.google.com/file/d/1kvCV8PxV6RCidehN4Wp78M116oZ_mSTX/view?usp=sharing)|
| pytorch QAT-INT8  | Histogram(MSE)  | 0.6961 | 0.5111 |-|-|[yolov7_qat.pt](https://drive.google.com/file/d/16Ylot5AfkjKeCyVlX3ECsuT6VmHULkd-/view?usp=sharing)|
| TensorRT FP16| -             | 0.6973 | 0.5124 |140 |168|[yolov7.onnx](https://drive.google.com/file/d/1R5muSJWVC_BQKml4s4wQQewUXdmQl0Mm/view?usp=sharing) |
| TensorRT PTQ-INT8 | TensorRT built in EntropyCalibratorV2 | 0.6317 | 0.4573 |207|264|-|
| TensorRT QAT-INT8 | Histogram(MSE)  | 0.6962 | 0.5113 |207|266|[yolov7_qat_640.onnx](https://drive.google.com/file/d/1qn-p4N3GZojIOvvxkzmPGCQKR6q4ov73/view?usp=sharing)|
 - network input resolution: 3x640x640
 - note: trtexec cudaGraph is enabled

## How To QAT Training 
### 1.Setup

For a pre-built environment that includes all necessary components, you may consider using the repository available at [levipereira/docker_images](https://github.com/levipereira/docker_images/tree/master/yolov7).

It deploys YOLOv7 with YOLO Quantization-Aware Training (QAT) patched. It also installs the TensorRT Engine Explorer (TREx), which is a Python library and a set of Jupyter notebooks for exploring a TensorRT engine plan and its associated inference profiling data.

Note: The pre-built environment mentioned above utilizes the `NVIDIA PyTorch image (nvcr.io/nvidia/pytorch:23.02-py3)`, which supports the latest GPUs such as Ada Lovelace/Hopper architecture.

 
### 2. Start QAT training
  ```bash
  ## Download yolov7 Model .
  wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt

  $ python3 scripts/qat.py quantize yolov7.pt --ptq=ptq.pt --qat=qat.pt --eval-ptq --eval-origin
  
  ## or custom dataset example
  $ python3 scripts/qat.py quantize yolov7.pt \
   --data data/custom.yaml \
   --hyp data/hyp.scratch.custom.yaml \
   --img-size 640 \
   --ptq=ptq.pt \
   --qat=qat.pt \
   --eval-ptq \
   --eval-origin \
   --use-pycocotools 

  ```
Output of Quantizing the COCO Dataset Using yolov7.pt: [Quantized yolov7 Model](doc/output_tests/quantize_yolov7_model_coco_output.txt)

  This script includes steps below: 
  - Insert Q&DQ nodes to get fake-quant pytorch model<br>
  [Pytorch quntization tool](https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization) provides automatic insertion of QDQ function. But for yolov7 model, it can not get the same performance as PTQ, because in Explicit mode(QAT mode), TensorRT will henceforth refer Q/DQ nodes' placement to restrict the precision of the model. Some of the automatic added Q&DQ nodes can not be fused with other layers which will cause some extra useless precision convertion. In our script, We find Some rules and restrictions for yolov7, QDQ nodes are automatically analyzed and configured in a rule-based manner, ensuring that they are optimal under TensorRT. Ensuring that all nodes are running INT8(confirmed with tool:[trt-engine-explorer](https://github.com/NVIDIA/TensorRT/tree/main/tools/experimental/trt-engine-explorer), see [scripts/draw-engine.py](./scripts/draw-engine.py)). for details of this part, please refer [quantization/rules.py](./quantization/rules.py), About the guidance of Q&DQ insert, please refer [Guidance_of_QAT_performance_optimization](./doc/Guidance_of_QAT_performance_optimization.md)

  - PTQ calibration<br>
  After inserting Q&DQ nodes, we recommend to run PTQ-Calibration first. Per experiments, `Histogram(MSE)` is the best PTQ calibration method for yolov7.
  Note: if you are satisfied with PTQ result, you could also skip QAT.
  
  - QAT training<br>
  After QAT, need to finetune traning our model. after getting the accuracy we are satisfied, Saving the weights to files

### 3. Export onnx 
  ```bash
  $ python scripts/qat.py export qat.pt --img-size=640 --save=qat.onnx --dynamic

  ##  exporting end2end

  $ python scripts/qat.py export qat.pt \
  --save=qat.onnx \
  --dynamic \
  --img-size 640 \
  --end2end \
  --topk-all 100 \
  --simplify \
  --iou-thres 0.45 \
  --conf-thres 0.25
  ```

### 4. Evaluate model accuracy on coco 
  ```bash
  $ bash scripts/eval-trt.sh  <weight_file .pt > <data/coco.yaml> <experiment_dir>
  
  ## example
  $ bash scripts/eval-trt.sh qat_best_5108.pt data/coco.yaml experiments/eval_trt

  
  ```
Output from Evaluating the COCO Dataset Using qat.pt: [Evaluation of TRT Engine with yolov7 Model](doc/output_tests/trt_eval_yolov7_coco.txt)

### 5. Benchmark
  ```bash
  $ /usr/src/tensorrt/bin/trtexec \
  --onnx=qat.onnx \
  --int8 \
  --fp16  \
  --workspace=1024000 \
  --minShapes=images:4x3x640x640 \
  --optShapes=images:4x3x640x640 \
  --maxShapes=images:4x3x640x640
  
  ```


## Quantization Yolov7-Tiny
```bash
$ python scripts/qat.py quantize yolov7-tiny.pt \
--qat=qat.pt \
--ptq=ptq.pt \
--ignore-policy="model\.77\.m\.(.*)|model\.0\.(.*)" \
--supervision-stride=1 \
--eval-ptq \
--eval-origin
```

## Note
- For YoloV5, please use the script `scripts/qat-yolov5.py`. This adds QAT support for `Add operator`, making it more performant.
- Please refer to the `quantize.replace_bottleneck_forward` function to handle the `Add operator`.