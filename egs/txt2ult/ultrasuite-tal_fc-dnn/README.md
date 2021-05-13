Download Merlin & txt2ult
---------------

Step 1: git clone https://github.com/BME-SmartLab/txt2ult/ 

Install tools
-------------

Similarly to original Merlin

Single speaker training
-----------------------

Please follow below steps:
 
Step 2: cd txt2ult/egs/txt2ult/ultrasuite-tal_fc-dnn/ <br/>
Step 3: ./run_full_voice.sh <UltraSuite-TaL dir> <speaker> <br/>
e.g. ./run_full_voice.sh ~/UltraSuite-TaL/TaL80/core/ 01fi 

Generate new sentences
----------------------

To generate new sentences, please follow below steps:

Step 4: ./08_merlin_synthesis.sh <speaker>  <br/>
Step 5: ./09_generate_ult_video.sh <speaker> 

Citation
--------

If you publish work based on Merlin & txt2ult, please cite: 

Implementation of Tamás Gábor Csapó, ,,Extending Text-to-Speech Synthesis with Articulatory Movement Prediction using Ultrasound Tongue Imaging'', submitted to SSW11, 2021.