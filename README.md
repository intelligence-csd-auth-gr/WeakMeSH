# WE-MINT-MUL

This repository contains the code for our [AIME 2021] (http://aime21.aimedicine.info/index.php) paper with title: A multi-instance multi-label weakly supervised approach for dealing with emerging MeSH descriptors


## Files
The link to the **Previous Host dataset** can be found here: https://drive.google.com/drive/folders/1mKwoPVOwnZXBgDeuZYwtxZV-SyTCKJgK?usp=sharing

The link to the **test set** containing all articles found for the top 100 labels can be found here: https://drive.google.com/drive/folders/1Zcyo_HuO94ToJAGIDGKOkXzZttZqbCT8?usp=sharing

The link to the **Complete results** including those shown inside the paper can be found here: https://drive.google.com/drive/folders/1NB6um2SA9Vwb-nbO7ggvMoR04dbC50KI?usp=sharing

The folder **create PH dataset** contains the files and code needed in order to create the Previous Host Dataset from the MeSH 2018 dataset.

The folder **important files** contains all the files needed to run the program. Specifically all the .txt files with the different sets of labels and the .pickle with the thresholds calculated by the GMMs approach.

The folder **other WSL approaches** has the code needed to transform our data to run the approaches we used to compare our method.

**GMMS.ipynb** has the code for calcuating the GMMs thresholds for each one of our labels

**measure_freqs.py** calculates how many times each label appears inside the test set

**mesh_multi_label_experiments.py** contains the code for running our approach.


## Configure

The requirements to run this program can be found in **requirements.txt**

For **mesh_multi_label_experiments.py** 

>set user = 'other' (line 631)

>set main_path to your desired absolute path (line 635). Said path must contain the files from **important files** folder as well as the **Previous Host Dataset** and **test set**.

 

## Run

The program was developed in Python 3.7

Example call:

`python3.7 mesh_multi_label_experiments.py`

Results will be stored at **main_path**:

>where main_path is an absolute path provided during configuration



