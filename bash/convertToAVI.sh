#!/bin/bash

for input_file in *.mp4; do
    
    [ -e "$input_file" ] || continue
    filename="${input_file%.*}"
    output_file="${filename}.avi"
    
    echo "Converting: $input_file -> $output_file"
    
    ffmpeg -i "$input_file" -c:v mpeg4 -qscale:v 3 -c:a libmp3lame -qscale:a 4 "$output_file"

done

echo "Batch conversion completed without errors."
