#!/bin/bash

orig_dir="Original MP4"
conv_dir="Converted AVI"
mkdir -p "$orig_dir"
mkdir -p "$conv_dir"

for input_file in *.mp4; do
    
    [ -e "$input_file" ] || continue
    filename="${input_file%.*}"
    output_file="${filename}.avi"
    
    echo "Processing $input_file -> $output_file"
    if ffmpeg -i "$input_file" -c:v mjpeg -qscale:v 2 -an "$conv_dir/$output_file"; then
        mv "$input_file" "$orig_dir/"
        
    else
        echo "Error: Failed to convert '$input_file'."
    fi

done
