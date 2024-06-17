#!/bin/sh
if [ -d "./OpenYOLO3D" ]
then
    echo "OpenYOLO3D already exists!"
else
    echo "Downloading OpenYOLO3D.zip ..."
    gdown 1FneLaYaClWDO51L9lIvlTQbheh5SfOFD
    echo "Unzipping OpenYOLO3D.zip to OpenYOLO3D..."
    unzip OpenYOLO3D.zip
fi
mkdir data/
echo "Moving OpenYOLO3D/replica to data/replica ..."
mv ./OpenYOLO3D/replica ./data/


