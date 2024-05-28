# Johnny Still Can’t Opt-out: Assessing the IAB CCPA Compliance Framework?



This repository contains the data and code to produce figures and analysis that appeared in the paper "Johnny Still Can’t Opt-out: Assessing the IAB CCPA Compliance Framework?" by Muhammad Abu Bakar Aziz and Christo Wilson, accepted at the Proceedings of the Privacy Enhancing Technologies Symposium (PoPETS) 2024. In this study, we take a deep dive into the IAB CCPA Compliance Framework to measure end-to-end flows of consent information and better understand why opt-out signals are not being honored.

## Data
 Download the dataset using this [link](https://personalization.ccs.neu.edu/static/archive/data.zip). This will download `data.zip` which is about 2.5 GB. 
 Then, unzip the data in the same repository folder:
```
unzip data.zip
```
This will create ```data``` folder that contains data used in the paper. It consists of about 32 GB.

## Script
This repository contains one main script, `figures.py` that can be used to plot all the figures, and tables, and analysis in the paper.

To produce all the figures and tables and analysis, run the following command:

```
python3 figures.py -f generate_fig_tables
```
This will create a ```fig```folder containing all the figures (1-10) in the paper, as well as tables 2-6 that appear in our paper.

To produce additional analysis. 
```
python3 figures.py -f generate_analysis > analysis
```

## Software Requirements
Our scripts were tested on Ubuntu 22.04.4 LTS and used Python 3.10.12.We expect it should work any Python version >= 3.8 and <=3.11. Our script also also used the following libraries:
```
pandas>=1.4.3
seaborn>=0.13.0
matplotlib>=3.5.2
numpy>=1.23.0
cycler>=0.11.0
scipy>=1.8.1
tldextract>=3.3.0
```

## Hardware Requirements.
These artifacts were tested on our lab cluster with Ubuntu 22.04.4 LTS, 189 GB RAM, and 32 cores. However, we expect the artifact to run on the following hardware:
```
Any good processor (I5 or I7 processor)
32 GB Ram
40 GB Disk Space
Good Internet Connection to download the dataset
```


## Steps to Reproduce Figures
1) git clone the repo `https://github.com/abubakaraziz/Assessing-IAB-CCPA-Framework`.
2) Download the data from the link above.
3) Then, unzip the data in the same repository folder:
   ```
   unzip data.zip
   ```
4) Install dependencies using ```pip3 install -r requirements.txt```.
5) Run ```python3 generate_fig_tables -f generate_fig_tables```

We believe the code should work without creating any virtual environment if you use python 3.10.12 version (and other latest version of python should work too). However, if you want or there are some issues, feel free to create python virtual environment by following these instructions [here](https://docs.python.org/3/library/venv.html). 

Create a virtual environment by running the following command in the repository directory using python version 3.10.12:
```
python3 -m venv iab_venv
```
This will create a directory named `iab_venv` in the current directory, containing the virtual environment.
Activate the virtual environment by running:
```
source iab/bin/activate
```
Once the virtual environment is activated, you can install dependencies using `pip3`. For example:
```
pip3 install -r requirements.txt
```
## Citation
If you use this code or data, please cite our paper. In Bibtex format:
```
@article{aziz-2024-pets,
  author = {Muhammad Abu Bakar Aziz and Christo Wilson},
  title = {{Johnny Still Can’t Opt-out: Assessing the IAB CCPA Compliance Framework?}},
  journal = {{Proceedings on Privacy Enhancing Technologies (PoPETS)}},
  volume = {2024},
  number = {4},
  month = {May},
  year = {2024},
}
```

