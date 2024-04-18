#!/usr/bin/env bash

set -e

if [ ! -f data/mscoco2017val/readme.txt ]; then

   echo "Downloading MS COCO validation 2017"
   mkdir -p data/mscoco2017val
   pushd data/mscoco2017val

   echo "Downloading Annotations"
   wget --progress=bar http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O annotations_trainval2017.zip
   unzip annotations_trainval2017.zip
   rm annotations_trainval2017.zip

   echo "Downloading Data"
   wget --progress=bar http://images.cocodataset.org/zips/val2017.zip -O val2017.zip
   unzip val2017.zip
   rm val2017.zip

   echo "MS COCO val2017: https://cocodataset.org/" >> readme.txt
   popd

fi
