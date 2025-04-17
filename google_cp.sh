#!/bin/bash

# Loop from 1 to 469
for i in $(seq -w 1 469); do
    src="data/preprocessed/google/${i}_ml.bin"
    dest="data/preprocessed/google/movement_${i}_ml.bin"

    if [[ -f "$src" ]]; then
        mv "$src" "$dest"
        echo "Renamed $src to $dest"
    else
        echo "File $src not found, skipping."
    fi
done

