#!/bin/bash

# Check if all required arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <weight_file> <dataset.yaml> <experiment_dir>"
    exit 1
fi

weight=$1
data=$2
experiment_dir=$3

project_name="eval_trt"

# Extract filename without path and extension
prefix=$(basename -- "$weight")
prefix="${prefix%.*}"

# Check if weight file exists
if [ ! -f "$weight" ]; then
    echo "Error: Weight file '$weight' not found."
    exit 1
fi

# Run the script
python3 scripts/qat.py export "$weight" --experiment "$experiment_dir" --project-name "" --dynamic --save="${prefix}.onnx" --img-size=672

echo -e "\n"

# Check if ONNX file was successfully generated
if [ ! -f "${experiment_dir}/${prefix}.onnx" ]; then
    echo "Error: ONNX file '${experiment_dir}/${prefix}.onnx' not generated."
    exit 1
fi

# Run the script
 bash scripts/generate_trt_engine.sh ${experiment_dir}/${prefix}.onnx $experiment_dir

# Check if Graph file exists
if [ ! -f "${experiment_dir}/${prefix}.graph" ]; then
    echo "Error: Graph file '${experiment_dir}/${prefix}.graph' not found."
    exit 1
fi

# Run the script
/bin/bash -c "source /opt/nvidia_trex/env_trex/bin/activate && python3 scripts/draw-engine.py --layer ${experiment_dir}/${prefix}.graph "  

# Run the script
python scripts/eval-trt.py --engine=${experiment_dir}/${prefix}.engine --data $data --img-size 640 --save-json 
