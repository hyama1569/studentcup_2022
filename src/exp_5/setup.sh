#!/bin/bash

#pip install wandb
#wandb login
pip install transformers
pip install seaborn
#pip install setuptools==59.5.0
#pip install -U spacy
#pip install hydra-core --upgrade
#pip install pytorch-lightning
#git clone https://github.com/SkafteNicki/pl_cross.git
#python ./pl_cross/setup.py install

#pip install torch==1.9+cu111 -f https://download.pytorch.org/whl/torch_stable.html


#data
#train.csv
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1EyKynZkPXVOqN-mpjsJcWdwIRwc25qk_' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1EyKynZkPXVOqN-mpjsJcWdwIRwc25qk_" -O train.csv && rm -rf /tmp/cookies.txt
#test.csv
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1nzxflpOH4xPGZnCNo-pfGtZ-NDue7b1a' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1nzxflpOH4xPGZnCNo-pfGtZ-NDue7b1a" -O test.csv && rm -rf /tmp/cookies.txt
#submit_sample.csv
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1w9ZwhbDMmqdCz8ixFO5PLcjZwst6KkSM' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1w9ZwhbDMmqdCz8ixFO5PLcjZwst6KkSM" -O submit_sample.csv && rm -rf /tmp/cookies.txt
#train.pickle
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1EyKynZkPXVOqN-mpjsJcWdwIRwc25qk_' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1EyKynZkPXVOqN-mpjsJcWdwIRwc25qk_" -O train.pickle && rm -rf /tmp/cookies.txt
#test.pickle
#wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1EyKynZkPXVOqN-mpjsJcWdwIRwc25qk_' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1EyKynZkPXVOqN-mpjsJcWdwIRwc25qk_" -O test.pickle && rm -rf /tmp/cookies.txt



#move files
mv t* ../../data
mv submit_sample.csv ../../data
