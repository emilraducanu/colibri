def find_download_directory():
    """Quick description

    Long description

    Parameters:
    query (str):

    Returns:

    """

    import os

    if os.name == "nt":  # Windows
        download_dirs = [os.getenv("DOWNLOAD_DIR")]
    else:  # Unix-like
        download_dirs = [
            os.path.expanduser("~/Downloads"),  # Fallback for Unix-like systems
            os.getenv("XDG_DOWNLOAD_DIR"),  # XDG user directory (Linux)
            os.getenv("DOWNLOAD_DIR"),  # Custom environment variable (if set)
        ]

    for dir_path in download_dirs:
        if dir_path and os.path.isdir(dir_path):
            return dir_path
    return None


def merger_cleaner(data_dir):
    """Quick description

    Long description

    Parameters:
    query (str):

    Returns:

    """

    import os
    import pandas as pd
    from src.glob_var import PLATFORM_MAP

    platform_plotted = []
    data = []
    for platform in os.listdir(data_dir):
        platform_plotted.append(str(platform))
        file_path = os.path.join(data_dir, platform, "data.pkl")
        df = pd.read_pickle(file_path)
        df = df.loc[:, PLATFORM_MAP.get(platform)[2].keys()]
        df = df.rename(columns=PLATFORM_MAP.get(platform)[2])
        df.insert(0, "Platform", str(platform))
        data.append(df)

    df = pd.concat(data)
    size_before = len(df)
    print(f"{size_before} papers merged into in a single DataFrame.")

    check_emptyness = ["DOI", "Title", "Abstract"]
    df = df.dropna(subset=check_emptyness, how="all")
    df = df.drop_duplicates()
    df = df[df["DOI"].duplicated(keep=False) == False]
    size_after = len(df)
    size_diff = size_before - size_after
    proportion = round(size_diff * 100 / size_before, 1)

    print(
        f"{size_diff} papers deleted ({proportion}%) because of lack of information or duplicates."
    )

    merged_cleaned_pub_dir = data_dir.replace("scrapped_pub", "merged_cleaned_pub")
    os.makedirs(merged_cleaned_pub_dir)
    merge_clean_file = os.path.join(merged_cleaned_pub_dir, "data.pkl")
    df.to_pickle(merge_clean_file)

    print(
        f"Data coming from the selected platform(s) are cleaned and merged into file {merge_clean_file} ."
    )
    print("Data ready to be passed into phase II.")

    return merge_clean_file


def wos(data_dir, query: str):
    """Get data (title, authors, DOI, etc.) of papers coming from Web of Science

    This function provides data of scientific articles coming from the results of a search query on Web of Science Core Collection (https://www.webofscience.com/wos/woscc/advanced-search).
    The data collected for each result are:
    * DOI
    * Title
    * Authors
    * Publication year
    * Abstract
    * Keywords
    * Language
    * Affiliation
    * Journal
    * Publisher
    * Volume
    * Issue
    Must be connected to a network that has access to Web of Science Core Collection.
    Must have Mozilla Firefox as a default web browser.

    Parameters:
    query (str): search query. Sythax (https://webofscience.help.clarivate.com/en-us/Content/wos-core-collection/woscc-search-field-tags.htm).

    Returns:
    The path of Pickle file containing data.
    """

    import os
    import shutil
    import time
    import pandas as pd
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.firefox.options import Options

    tmp_xls_files = os.path.join(data_dir, "tmp")
    os.makedirs(tmp_xls_files)

    url = "https://www.webofscience.com/wos/woscc/advanced-search"
    firefox_options = Options()
    firefox_options.add_argument("-headless")
    driver = webdriver.Firefox(options=firefox_options)
    driver.get(url)

    print(f"Connected to Web of Science.")

    time.sleep(5)
    driver.find_element(By.ID, "onetrust-reject-all-handler").click()

    time.sleep(5)
    search_input = driver.find_element(By.ID, "advancedSearchInputArea")
    search_input.send_keys(query)
    search_input.submit()

    time.sleep(5)
    nb_results = driver.find_element(
        By.XPATH,
        "/html/body/app-wos/main/div/div/div[2]/div/div/div[2]/app-input-route/app-base-summary-component/app-search-friendly-display/div[1]/app-general-search-friendly-display/div[1]/div[1]/h1/span",
    ).text
    nb_results = int(nb_results.replace(",", ""))

    print(f"Found {nb_results} papers.")

    def downloader(input_aera_index, sent_key_low, sent_key_high):
        """Quick description

        Long description

        Parameters:
        query (str):

        Returns:

        """

        time.sleep(5)
        driver.find_element(
            By.XPATH,
            "/html/body/app-wos/main/div/div/div[2]/div/div/div[2]/app-input-route/app-base-summary-component/div/div[2]/app-page-controls[1]/div/app-export-option/div/app-export-menu/div/button",
        ).click()

        time.sleep(1)
        driver.find_element(By.ID, "exportToExcelButton").click()

        time.sleep(1)
        records_from_button = driver.find_element(By.ID, "radio3-input")
        driver.execute_script("arguments[0].click();", records_from_button)

        low_bound = driver.find_element(
            By.ID, "mat-input-" + str(2 * input_aera_index + 1)
        )
        low_bound.clear()
        low_bound.send_keys(str(sent_key_low))
        high_bound = driver.find_element(
            By.ID, "mat-input-" + str(2 * input_aera_index + 2)
        )
        high_bound.clear()
        high_bound.send_keys(str(sent_key_high))

        driver.find_element(
            By.XPATH,
            "/html/body/app-wos/main/div/div/div[2]/div/div/div[2]/app-input-route[1]/app-export-overlay/div/div[3]/div[2]/app-export-out-details/div/div[2]/form/div/div[1]/wos-select/button",
        ).click()
        driver.find_element(
            By.XPATH,
            "/html/body/app-wos/main/div/div/div[2]/div/div/div[2]/app-input-route[1]/app-export-overlay/div/div[3]/div[2]/app-export-out-details/div/div[2]/form/div/div[1]/wos-select/div/div/div/div[3]",
        ).click()
        driver.find_element(
            By.XPATH,
            "/html/body/app-wos/main/div/div/div[2]/div/div/div[2]/app-input-route[1]/app-export-overlay/div/div[3]/div[2]/app-export-out-details/div/div[2]/form/div/div[2]/button[1]",
        ).click()

        time.sleep(10)
        download_directory = find_download_directory()
        if not download_directory:
            print("Could not determine the download directory.")
            return 0

        full_path_downloaded_files = []
        for filename in os.listdir(download_directory):
            if filename.startswith("savedrecs") and filename.endswith(".xls"):
                full_path_downloaded_files.append(download_directory + "/" + filename)

        latest_file = max(full_path_downloaded_files, key=os.path.getctime)
        shutil.move(
            latest_file, os.path.join(tmp_xls_files, str(input_aera_index + 1) + ".xls")
        )

    for batch_nb in range(int(int(nb_results) / 1000)):
        downloader(batch_nb, int(batch_nb * 1000) + 1, int(batch_nb * 1000) + 1000)
        downloaded_count = (batch_nb + 1) * 1000
        print(f"{downloaded_count} papers downloaded over {nb_results}.")
    downloader(
        int(int(nb_results) / 1000), int(int(nb_results) / 1000) * 1000, nb_results
    )
    print(f"{nb_results} papers downloaded over {nb_results}.")

    driver.quit()

    data = []
    for filename in os.listdir(tmp_xls_files):
        file_path = os.path.join(tmp_xls_files, filename)
        df = pd.read_excel(file_path)
        data.append(df)
    df = pd.concat(data)
    size = len(df)

    shutil.rmtree(tmp_xls_files)

    wos_folder = os.path.join(data_dir, "WoS")
    os.makedirs(wos_folder)
    wos_file = os.path.join(wos_folder, "data.pkl")
    df.to_pickle(wos_file)
    print(
        f"\nRaw data of {size} papers from Web of Science Core Collection saved to {wos_file}."
    )

    return wos_file


def scrape(query: str, platforms: list[str]):
    """Get publications from platforms specified

    Scrape data from the results of a search query. Scrapping will be performed on each platform specified. Data will be stored in 'colibri/data'.

    Parameters:
    query (str): universal search query that will be used in each database platform specified in parameters.
    platforms (list[str]): list of platforms that will be scrapped. Be careful of the sythax. Supported: ["WoS"]

    Returns:
    None
    """
    from src.glob_var import PLATFORM_MAP

    if platforms == []:
        print(
            "Parameter 'platforms' is empty. At least one platform must be specified."
        )
    else:
        valid_platforms = []
        for platform in platforms:
            if platform in PLATFORM_MAP:
                valid_platforms.append(platform)
            else:
                print(
                    f"'{platform}' is passed as a platform but not supported by colibri. Correct and re-run the function."
                )
                return 0

        import pytz
        import datetime
        import os

        platform_plotted = []
        for platform in valid_platforms:
            platform_plotted.append(str(PLATFORM_MAP.get(platform)[0]))

        sentence_template = "You selected platform(s) {} to get the publications from."
        formatted_sentence = sentence_template.format(", ".join(platform_plotted))
        print(formatted_sentence)

        utc_current_time = datetime.datetime.now(tz=pytz.timezone("UTC"))
        utc_current_time_str = (
            str(utc_current_time.year)
            + "-"
            + str(utc_current_time.month)
            + "-"
            + str(utc_current_time.day)
            + "_"
            + str(utc_current_time.hour)
            + "-"
            + str(utc_current_time.minute)
            + "-"
            + str(utc_current_time.second)
        )

        print(f"\nScrapping started at {utc_current_time_str} UTC.")

        current_dir = os.getcwd()
        target_folder = "colibri"
        while os.path.basename(current_dir) != target_folder:
            current_dir = os.path.dirname(current_dir)
        data_dir = os.path.join(
            current_dir, "data/scrapped_pub/" + utc_current_time_str
        )
        os.makedirs(data_dir)

        for platform in valid_platforms:
            PLATFORM_MAP.get(platform)[1](data_dir, query)

    return data_dir


def scrapping_over_time():
    """Quick description

    Long description

    Parameters:
    query (str):

    Returns:

    """

    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import os

    current_dir = os.getcwd()
    target_folder = "colibri"
    while os.path.basename(current_dir) != target_folder:
        current_dir = os.path.dirname(current_dir)

    dates = []
    sizes = []
    data_dir = os.path.join(current_dir, "data/merged_cleaned_pub")
    for subdir in os.listdir(data_dir):
        file_path = os.path.join(data_dir, subdir, "data.pkl")
        df = pd.read_pickle(file_path)
        date = subdir.split("_")[0]
        dates.append(date)
        sizes.append(len(df))

    data = {"Date": dates, "Number of publications": sizes}
    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by="Date")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=df["Date"],
        y=df["Number of publications"],
        hue=df["Number of publications"],
        palette="plasma",
        marker="o",
    )

    plt.title(
        "Number of publications scrapped on WoS Core Collection over time\n(cf. search query)"
    )
    plt.xlabel("Date")
    plt.ylabel("Number of publications")
    plt.xticks(ticks=df["Date"], labels=df["Date"].dt.strftime("%Y-%m-%d"), rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, "visualisations/scrapping_over_time.png"))
