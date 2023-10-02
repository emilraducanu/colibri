# PACKAGING FEATURES

For each module `scrapper.py`, `filter.py`, `characteriser.py` and `wrapper.py`:

- For each function, specify type of variables and type of return
- For each function, import only what is needed
- For each function, write Docstrings
- Update `environment.yml`
- Update `__init__.py`

# OPERATIONAL FEATURES

1. Test the project on non-Unix OS
2. Support other Web browsers to scrape Web of Science
3. Scrape other sources to get the publications
4. Create a unified search query for all scrapping sources
5. Create a robust benchmark for text classification models, with a dummy model as reference
6. Optimising DistilBERT F1-score
7. Implement step "3. Characterising", in order to fill the [output file](./data/template_output_database/doc.txt) in a semi-automated way
8. Publication language checking (traduction?)
9. Migrate the project management on `hatch`
