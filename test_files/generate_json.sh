#!/bin/bash

# simple script to generate json file with base64 encoded jpg image param

if (( $# < 1 ))
then
    echo "Usage: $0 IMAGE_FILE [OUTPUT_FILE]"
    exit 1
fi

if (( $# == 1 ))
then
    OUTPUT=${1/.*/.json}
else
    OUTPUT=$2
fi
BASE64_IMAGE=$(base64 -i $1 | xargs echo "data:image/jpeg;base64," | sed "s/ //" )
echo "{ \"image\": \"${BASE64_IMAGE}\" }" > "$OUTPUT"

