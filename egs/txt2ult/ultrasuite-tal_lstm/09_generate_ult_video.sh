#!/bin/bash -e


if test "$#" -ne 1; then
    echo "################################"
    echo "Usage:"
    echo "./09_generate_ult_video.sh <speaker>"
    echo ""
    echo "Default path to acoustic conf file: conf/acoustic_${Voice}.conf"
    echo "################################"
    exit 1
fi

speaker=$1

global_config_file=conf/global_settings_${speaker}.cfg
source $global_config_file


### 
echo "generating ultrasound video..."
python3 scripts/generate_ult_video.py $speaker


