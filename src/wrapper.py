def run_pipeline(query: str, platforms: list[str]):
    """Run the whole colibri pipeline

    Wrapper function that will call several sub-functions to execute the whole colibri pipeline, from the scrapping to the characterisation. More details of the pipeline in the README.md file.

    Parameters:
    query (str): universal search query that will be used in each database platform specified in parameters.
    platforms (list[str]): list of platforms that will be scrapped. Be careful of the sythax. Supported: ["WoS"]

    Returns:
    None
    """
    print("\033[1m\U0001F426 Welcome to the pipeline of colibri!\033[0m")

    import sys

    sys.path.append("..")
    import src

    # /!\ To be completed when new platforms supported
    PLATFORM_MAP = {
        "WoS": [
            "Web of Science Core Collection",
            src.scrapper.wos,
            {
                "DOI": "DOI",
                "Article Title": "Title",
                "Abstract": "Abstract",
                "Author Keywords": "Keywords",
            },
        ]
    }

    print("\n\033[1mPhase I - Scrapping\033[0m")
    scrapped_pub_data_dir = src.scrapper.scrape(query, platforms, PLATFORM_MAP)
    merged_cleaned_pub_dir = src.scrapper.merger_cleaner(
        scrapped_pub_data_dir, PLATFORM_MAP
    )

    print("\n\033[1mPhase II - Filtering\033[0m")
