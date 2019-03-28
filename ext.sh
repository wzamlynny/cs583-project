#!/bin/sh

for f in data/*.zip; do
    echo $f
    DIR=./data/$(basename "$f"| cut -f 1 -d '.')
    mkdir $DIR
    unzip -d $DIR $f
done

