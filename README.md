# RecSys Challenge 2023: UMONS Team solution

This repository contains a solution to the RecSys Challenge 2023: http://www.recsyschallenge.com/2023/

## 1. Prepare the environment

The file "environment.yml" contains all the libraries used for this challenge.
It can be used to recreate a similar environment: 

```
conda env create -f environment.yml
conda activate recsys2023
```

If you don't want to use conda, install the libraries that are imported in the notebook (refer to "environment.yml" for versions).

### Requirements for GPU training

cuda version: 11.2.2 <br/>
cudnn version: 8.1.1

## 2. Prepare the data folder

Extract the data zip file at the root of this repository and rename the folder as data if necessary.
Data folder should contain the directories "train" and "test".

## 3. Train the model

Run the notebook to train the model.
Alternatively, the notebook has been exported as a python script that you can run:

```
python train_model.py
```

## 4. Note on seeds

Seeds are used to get the same results at each launch of the notebook/script . However, when training with GPU, other sources of randomness seem to appear. Results may differ (not significantly) from one training to another one.

## 5. Authors

Maxime Manderlier, University of Mons (UMONS), Belgium (maxime.manderlier@umons.ac.be) <br/>
Fabian Lecron, University of Mons (UMONS), Belgium (fabian.lecron@umons.ac.be)