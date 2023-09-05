![Logo](./logos/banner_colibri.png)
---
`colibri` is a Python package aimed to synthesise scientific literature, using different Machine Learning techniques. Specify the topic you want to study, `colibri` will select publications from various sources and will analyse, extract and compile relevant data from them.

This project was developed as part of the [Operationalising International Research Cooperation on Soil carbon (ORCaSa)](https://irc-orcasa.eu/), an initiative that aims to bring together international stakeholders working on techniques for capturing and storing carbon in the soil.

:warning: *Important note* <br/>
`colibri` *is developed in the spirit of synthesising all kinds of publications focusing on any topic. For the moment, only the compilation of meta-analyses dealing with the impact of human practices on soil organic carbon is currently implemented.*

<br/>

# 🎬 Pipeline backstage
The data passes through three major and independant steps to get the final `colibri`'s output. Here is some details about these chained blocks.
### 1. Scrapping
xxx

### 2. Screening
xxx

### 3. Characterising
xxx

<br/>

# 🚦 Getting started
### 1. Prerequisites
Before setting up the environment, make sure you have `conda` installed on your system. If not, you can download and install it from the following links:

- [Anaconda](https://docs.anaconda.com/free/anaconda/install/) (used by the developers for this project)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

To properly use Selenium (automation of web browsing), you need to download the Driver corresponding to the default web browser of your system. To do so, you can follow the steps of this link:

- [Selenium Client Driver](https://www.selenium.dev/selenium/docs/api/py/#drivers)
### 2. Setup

To set up the environment and install the required dependencies, run the following commands in your working directory:
```bash
git clone https://github.com/emilraducanu/colibri.git
cd colibri
conda env create -f environment.yml
conda activate colibri
```

The package is now ready to use. You can test the different features in next the section.

<br/>

# 💡 Use cases
Simple use cases of the package can be found in [this](playground/playground.ipynb) Jupyter Notebook. You can run the the cells in your IDE or use the Jupyter server in your localhost by running the following command:
```bash
jupyter notebook
```
Make sure to follow steps of the **⚙️ Getting started** section beforehand.

<br/>

# 📝 To do
`colibri` is an open-source project still under development. Here is a non-exhaustive list of features that could be implemented in the future. Feel free to contribute!
1. Support other Web browsers to scrape Web of Science
2. Scrape other sources to get the publications
3. Create a unified search query for all scrapping sources
4. Create a robust benchmark for text classification models, with a dummy model as reference
5. Get PDFs of publications
and more...

<br/>

# 📜 Author & license
`colibri` is designed by [Emil Răducanu](https://github.com/emilraducanu) and [Damien Beillouin](https://github.com/dbeillouin).<br/>
This project is under license [GNU General Public License v3.0 or later](LICENSE.md).

<br/>

<p align="center">
  <img src=./logos/gplv3-with-text-84x42.png />
  <br/>
  <em>colibri Copyright © 2023 Emil Răducanu</em>
</p>
