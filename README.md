# Introduction
Hybrid recommendation based system that implements state-of-the-art NLP embedding methods to classify and suggest most related documents to users.


<p align="center"> <img src="https://user-images.githubusercontent.com/34713088/177145390-bd34f24a-856a-4c76-a267-b79c26ac6f0e.png" width="850" height="2100">


### Built With

The following technologies, frameworks and libraries have been used:

* [Python](https://www.python.org/)
* [Git](https://git-scm.com/)

We strongly suggest to create a virtual env (i.e. 'recommendation_system') providing the python version otherwise it will not install previous libraries:
```bash
conda create -n recommendation_system python=3.8.8 
conda activate recommendation_system
```

If you want to run it manually you need to have python 3.8.8 configured on your machine. 

1. Install all the libraries using the requirements.txt files that can be found in the main repository

```bash
pip install -r requirements.txt
```
2. Run the system

```bash
python main.py -ep "end_path" -p "path" -mp "model_path" -pdf "pdf_path" -u 'utent row number' -up "user_path" -c "category"
``` 
where:
* -ep folder path that will contain generated files
* -p folder path containing category folders
* -emb switch to specify whether to compute embedding of all documents
* -mp folder path that will contain model files, it can coincide with -ep path
* -pdf path to pdf file for recommendation
* -u user row number in database
* -up path containing user data
* -c category name

i.e.

```bash
python main.py -ep "C:\Desktop\recommendation_data" -p "C:\Desktop\Datasets" -emb "yes" -pdf "C:\Desktop\cool_file.pdf" -u 3000 -up "C:\Desktop\user_data.csv" -c "Fancy category"
``` 

# Possible next steps

1. Configure a docker registry in order to publish docker images 
2. Code refactoring to remove unused code and polish it
3. Use Cython for Doc2Vec embedding to allow parallelization and faster performance 
