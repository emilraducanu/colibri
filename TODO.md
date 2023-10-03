# PACKAGING FEATURES

For each module `scrapper.py`, `filter.py`, `characteriser.py` and `wrapper.py`:

- For each function, specify type of variables and type of return
- For each function, import only what is needed
- For each function, write Docstrings
- Update `environment.yml`
- Update `__init__.py`

# OPERATIONAL FEATURES

- Test the project on non-Unix OS
- Support other Web browsers to scrape Web of Science
- Scrape other sources to get the publications
- Implement a system to manage several search queries for each scrapping sources
- Create a robust benchmark for text classification models, with a dummy model as reference
- Optimising DistilBERT F1-score
- Step "3. Characterising" - fill the [output file](./data/template_output_database/doc.txt) in a semi-automated way
Publication language checking (traduction?)
- Migrate the project management on `hatch`
