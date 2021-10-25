# Trident Dehazing Forward

1- git clone https://github.com/MuhammetAkcann/TridentForward/

2- cd TridentForward

3- git clone https://github.com/jinfagang/DCNv2_latest 

4- cd DCNv2_latest 

5- sh make.sh (do not forget the python version in this file matches with the python you are using for pytorch env)

6- cd ..

7- python main.py (you can specify input file by changing line 84)

prerequrties:

pretrainedmodels==0.7.4

torchvision==0.9.1

torch==1.8.1

tqdm
