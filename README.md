# WeakMeSH

This repository contains the code for our [AIME 2021](http://aime21.aimedicine.info/index.php) paper with title: 

### A multi-instance multi-label weakly supervised approach for dealing with emerging MeSH descriptors

It also contains the code for the extension Published in [Artificial Intelligence in Medicine] journal titled: 
### WeakMeSH: Leveraging Provenance Information for Weakly Supervised Classification of Biomedical Articles with Emerging MeSH Descriptors

![complex_changes](https://user-images.githubusercontent.com/23103855/105720908-a49ec380-5f2c-11eb-86a8-eff2014f1941.jpg)

## Files
The link containing: the **Previous Host dataset** , the **test set** containing all the embeddings for the articles found for all 100 most-frequent labels, the **Complete results** including those shown inside the paper as well as the input files for WMIR can be found here: https://drive.google.com/drive/folders/1tE3glPshLIqb8wbO9r5K5u7ppceb5H1C?usp=sharing

The folder **create PH dataset** contains the files and code needed in order to create the Previous Host Dataset from the MeSH 2018 dataset.

The folder **important files** contains all the files needed to run the program. Specifically all the .txt files with the different sets of labels and the .pickle with the thresholds calculated by the GMMs approach.

The folder **other WSL approaches** has the code needed to transform our data to run the approaches we used to compare our method.

**GMMS.ipynb** has the code for calcuating the GMMs thresholds for each one of our labels

**measure_freqs.py** calculates how many times each label appears inside the test set

**mesh_multi_label_experiments.py** contains the code for running our approach.


## Configure

The requirements to run this program can be found in **requirements.txt**

For **mesh_multi_label_experiments.py** :

>set user = 'other' (line 631)

>set main_path to your desired absolute path (line 635). Said path must contain the files from **important files** folder as well as the **Previous Host Dataset** and **test set**.

For **find_seed_words.py** which can be found inside **create PH dataset folder**:

>set the arguement for method create_seed_word_list in line 67 to the .bin file of your choice depending on which year's descriptors you wish to get the info for

## Run

The program was developed in Python 3.7

Example call:

`python3.7 mesh_multi_label_experiments.py`

Results will be stored at **main_path**:

>where main_path is an absolute path provided during configuration

After executing the program you will be asked to choose the setting which you want to run

**1st** for how many labels you wish to execute the procedure: **10** which is a small example of our method using only the top 10 most frequent labels, **100** which runs our method for the top 100 most frequent labels (the labels without PI or PMN are ignored and thus have no train examples) and finally **62** which are the experiments shown in our paper.

**2nd** The process to be executed. The choice **0 (Raw data)** should be selected if it's the first time you run the program.

**3rd** The choice for weakly labeled threshold. Either a number betwee [0,1] or the choice for the automatically calculated thresholds by the GMMs.

**4th** The mode to be executed (**Choice 1** for the method proposed inside the paper, **Choice 2** for Prime, **Choice 3** for Extended Prime, **Choice 4** for our method without PI or PMN information, **Choice 5** for our ZSL Method discussed in another paper)

**5th** How many of the data from the PH dataset should be used to find the weakly-labeled train-set

## Developed by: 

|           Name  (English/Greek)            |      e-mail          |
| -------------------------------------------| ---------------------|
| Nikolaos Mylonas    (Νικόλαος Μυλωνάς)     | myloniko@csd.auth.gr |
| Stamatis Karlos     (Σταμάτης Κάρλος)      | stkarlos@csd.auth.gr |
| Grigorios Tsoumakas (Γρηγόριος Τσουμάκας)  | greg@csd.auth.gr     |

## Funded by

The research work was supported by the Hellenic Foundation forResearch and Innovation (H.F.R.I.) under the “First Call for H.F.R.I.Research Projects to support Faculty members and Researchers and the procurement of high-cost research equipment grant” (ProjectNumber: 514).

## Additional resources

- [AMULET project](https://www.linkedin.com/showcase/amulet-project/about/)
- [Academic Team's page](https://intelligence.csd.auth.gr/#)
 
 ![amulet-logo](https://user-images.githubusercontent.com/6009931/87019683-9204ad00-c1db-11ea-9394-855d1d3b41b3.png)



