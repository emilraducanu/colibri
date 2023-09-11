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

    from sys import path

    path.append("..")
    from src.scrapper import scrape, merger_cleaner

    print("\n\033[1mPhase I - Scrapping\033[0m")
    scrapped_pub_data_dir = scrape(query, platforms)
    merged_cleaned_pub_dir = merger_cleaner(scrapped_pub_data_dir)

    print("\n\033[1mPhase II - Filtering\033[0m")
