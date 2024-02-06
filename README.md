![](visualisations/logos/banner_colibri.png)
---
`colibri` is a Python project aimed to synthesise scientific literature. Specify the topic you want to study, `colibri` will select publications from various sources and will analyse, extract and compile relevant data from them.

This project was developed as part of the [Operationalising International Research Cooperation on Soil carbon (ORCaSa)](https://irc-orcasa.eu/), an initiative that aims to bring together international stakeholders working on techniques for capturing and storing carbon in the soil.

⚠️ *Important note* `<br/>`
`colibri` *is developed in the spirit of synthesising all kinds of publications focusing on any topic. For the moment, only the compilation of meta-analyses dealing with the impact of human practices on soil organic carbon is currently implemented.*

<br/>

# 🎬 Pipeline backstage

`colibri` aims to help scientists to conduct umbrella reviews, i.e. meta-analyses of meta-analyses, in order to get an live overview of the knowledge of a given field.`<br/>`
The data passes through three major and independant stages to get the final `colibri`'s output. Here is some details about these chained blocks.

### 1. Scrapping

Publications from diverse sources need to be collected, so scrapping functions are implemented for different online scientific litterature platforms. Each plateforms structure is different. This heterogeneity is taken into account while dealing with platforms' APIs or raw HTML code. The scrapping can be run at anytime to update the database. The data collected corresponds to the results of a specific search query.`<br/>`
Every publications are stored into in single DataFrame. At this point, a publication is characterised with four variables: DOI (acting as a unique ID key), title, abstract and keywords. The user can manually add new publications at this step. This DataFrame is then cleaned from duplicates and missing data and is ready to be passed into next phase.

### 2. Screening

The scrapped publications need to be classified between two sets: corresponding indeed to to field studied or not. To do so, the pre-trained binary classification model DistilBERT is used. The model is fine-tuned to make predictions on the field currently studied ("impact of human practices on soil organic carbon", for the moment). The [training set](./data/distilbert_trainset/trainset.csv) comes from [Beillouin, D., et al.](https://doi.org/10.1038/s41597-022-01318-1).`<br/>`
Once trained, publications can be classified with a certain level of confidence. The user can correct the predictions of the model at this step, by forcing the inclusion/exclusion of a publication. The filtered data is now ready to be passed into next phase.

### 3. Characterising

Raw PDFs of each publications is scrapped and then converted into text. Various variables are extracted from the text to be able to fill the [final output file](./data/template_colibri_output.json).

<br/>

# 🚦 Getting started

### 1. Prerequisites

Before setting up the environment, make sure you have `conda` installed on your system. If not, you can download and install it from the following links:

- [Anaconda](https://docs.anaconda.com/free/anaconda/install/) (used by the developers for this project)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

To automate web browsing, we use Selenium. You need to download the Driver corresponding to the default web browser of your system. To do so, you can follow the steps of this link:

- [Selenium Client Driver](https://www.selenium.dev/selenium/docs/api/py/#drivers)

### 2. Setup

To set up the environment and install the required dependencies, run the following commands in your working directory:

```bash
git clone https://github.com/emilraducanu/colibri.git
cd colibri
conda env create -f environment.yml
conda activate colibri
```

The project is now ready to use. You can test the different features in next the section.

<br/>

# 💡 Use cases

Simple use cases of the project can be found in [this](playground/playground.ipynb) Jupyter Notebook. You can run the the cells in your IDE or use the Jupyter server in your localhost by running the following command:

```bash
jupyter notebook
```

Make sure to follow steps of the **⚙️ Getting started** section beforehand.

<br/>

# 📝 To do

`colibri` is a FOSS project still under development. You will find [here](TODO.md) a non-exhaustive list of features that could be implemented in the future. Feel free to contribute!

<br/>

# 📜 Author & license

`colibri` is designed by [Emil Răducanu](https://github.com/emilraducanu) and [Damien Beillouin](https://github.com/dbeillouin).`<br/>`
This project is distributed under [GNU General Public License v3.0 or later](COPYING.md).

<br/>

<p align="center">
  <img src="visualisations/logos/gplv3-with-text-84x42.png"/>
  <br/>
  <em>colibri Copyright © 2023 Emil Răducanu</em>
</p>
