![Logo](./logos/banner_colibri.png)
---
**colibri** is a Python package aimed to synthesise scientific literature, using Natural Language Processing (NLP) techniques. Specify the topic you want to study, **colibri** will select publications from various sources and will analyse, extract and compile relevant data from them.

This project was developed as part of the [Operationalising International Research Cooperation on Soil carbon (ORCaSa)](https://irc-orcasa.eu/), an initiative that aims to bring together international stakeholders working on techniques for capturing and storing carbon in the soil.

:warning: *Important note* <br/>
**colibri** *is developed in the spirit of synthesising all kinds of publications focusing on any topic. For the moment, only the compilation of meta-analyses dealing with the impact of human practices on soil organic carbon is currently implemented.*

<br/>

# 🎬 Pipeline backstage
### 1. Scrapping
xxx

### 2. Filtering
xxx

### 3. Characterising
xxx

<br/>

# 🚦 Getting started
### 1. Prerequisites
Before setting up the environment, make sure you have `conda` installed on your system. If not, you can download and install it from the following links:

- [Anaconda](https://docs.anaconda.com/free/anaconda/install/) (used by the developers for this project)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### 2. Setup

To set up the environment and install the required dependencies, run the following commands:
```bash
conda env create -f env.yml
conda activate colibri
pip install -e .
```
To properly use Selenium (automation of web browsing), you need to download the Driver corresponding to the default web browser of your system. To do so, you can follow the steps of this link:

- [Selenium Client Driver](https://www.selenium.dev/selenium/docs/api/py/#drivers)

<br/>

The package is now ready to use. You can test the different features in next the section.

<br/>

# 💡 Use cases
Simple use cases of the package can be found in [this](playground/playground.ipynb) Jupyter Notebook. You can run the the cells in your IDE or use the jupyter server in your localhost by running the following command:
```bash
jupyter notebook
```
Make sure to follow steps of the **⚙️ Getting started** section beforehand.

<br/>

# 📝 To do
**colibri** is an open-source project still under development. Here is a non-exhaustive list of features that could be implemented in the future. Feel free to contribute!
1. Support other OS than Ubuntu.
2. Support other Web browsers than Mozilla Firefox to scrape Web of Science.

<br/>

# 📜 Author & license
This project is developed by [Emil Răducanu](https://github.com/emilraducanu).<br/>
License: to be determined.