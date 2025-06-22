# Predictive-Analysis-and-Forecasting-of-Commodity-Market-Dynamics

## Project Overview

This repository contains a comprehensive Jupyter Notebook along with a fully configured pipeline developed as part of my Bachelor’s Thesis Project for the International School of Economics (ISET), Class of 2025.

The `notebook.ipynb` in this repository includes all the relevant data processing, modeling, and explanatory commentary that supports the main findings of the thesis. This README serves as a technical guide and quickstart manual for setting up and running the pipeline. 

## Dataset Overivew

This project uses a synthetically generated daily commodity price dataset spanning from 2015 to 2025. The dataset includes typical financial features such as Open, High, Low, Close prices, Trading Volume, and a Market Sentiment Score (ranging from -1 to 1). It also incorporates macroeconomic indicators like GDP Growth, Inflation, and Interest Rates.

The data was sourced from Kaggle and simulated using Python to closely mimic real-world commodity market behaviors, including price volatility and sentiment-driven fluctuations. This realistic synthetic dataset is well-suited for predictive analysis and time series forecasting, benchmarking machine learning models, and exploring market dynamics.

[Dataset Link](https://www.kaggle.com/datasets/arittrabag/synthetic-commodity-price-data-2015-2025)

## Prerequisites

Before diving into the detailed steps of setting up and using this project, there are few important prerequisites or requirements that need to be addressed. These prerequisites ensure that your local development environment is ready and capable of efficiently running and supporting the project.

### Forking and Cloning from GitHub
Create a copy of this repository by forking it on GitHub.

Clone the forked repository to your local machine:

```bash
git clone https://github.com/Sandrog112/BA-Thesis.git
```

### Setting Up Development Environment
Ensure you have Python 3.10+ installed on your machine. 

Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

## Project structure:

This project has a modular structure, where each folder has a specific duty. Many of these files are gitignored and not visible in the repo.

```bash
Predictive-Analysis-and-Forecasting-of-Commodity-Market-Dynamics
├── data
│   └── raw.csv                 # Raw commodity price dataset
├── logs                        # Logs generated during training and inference
├── models                      # Folder for saved trained models
├── performance                 # Contains evaluation plots and predictions
├── src
│   ├── data_preprocessing.py   # Data preprocessing functions
│   ├── __init__.py
│   ├── train
│   │   ├── train.py            # Model training script
│   │   └── __init__.py
│   ├── inference
│   │   ├── inference.py        # Inference script
│   │   └── __init__.py
├── .gitignore
├── notebook.ipynb              # Main analysis and experimentation notebook
├── README.md                   # Project's instructions and overviews
├── requirements.txt            # Python dependencies

```

