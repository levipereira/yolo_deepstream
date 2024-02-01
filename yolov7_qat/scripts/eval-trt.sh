#!/bin/bash

# This script only worked on the Quantization model, otherwise, errors would throw up (Origin model, etc.)
weight=$1
prefix=${weight%.*}
onnx=${prefix}.onnx
graph=${prefix}.graph
engine=${prefix}.engine

# onnx must be 672x672 of input
python scripts/qat.py export $weight --dynamic --save=$onnx --size=672

# To obtain more QPS can add --fp16 flag for detect layer
trtexec --onnx=qat_fire.onnx \
    --saveEngine=qat_fire.engine --int8 --buildOnly --memPoolSize=workspace:1024MiB \
    --dumpLayerInfo --exportLayerInfo=qat_fire.graph --profilingVerbosity=detailed

#trtexec --onnx=$onnx --saveEngine=${engine} --int8 --buildOnly --memPoolSize=workspace:1024MiB  --dumpLayerInfo --exportLayerInfo=${graph} --profilingVerbosity=detailed

python scripts/draw-engine.py ${graph}
python scripts/eval-trt.py --engine=${engine} --data /dataset/data_road_fire_smoke.yaml --hyp hyp.scratch.fire_smoke.yaml
 