#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_onnx_file> <output_dir>"
    exit 1
fi

# Extract input file name without extension
input_file="$1"
file_name=$(basename -- "$input_file")
file_name_no_ext="${file_name%.*}"

# Output directory
output_dir="$2"

# Create output directory if it doesn't exist
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
    echo "Created output directory: $output_dir"
fi

# Generate engine and graph file paths
engine_file="${output_dir}/${file_name_no_ext}.engine"
graph_file="${output_dir}/${file_name_no_ext}.graph"

# Run trtexec command to generate engine and graph files
trtexec --onnx="${input_file}" \
        --saveEngine="${engine_file}" --f16 --int8 --buildOnly --memPoolSize=workspace:1024MiB \
        --dumpLayerInfo --exportLayerInfo="${graph_file}" --profilingVerbosity=detailed

# Check if trtexec command was successful
if [ $? -eq 0 ]; then
    echo "Engine and graph files generated successfully:"
    echo "Engine file: ${engine_file}"
    echo "Graph file: ${graph_file}"
else
    echo "Failed to generate engine and graph files."
fi
