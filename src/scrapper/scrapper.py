# Homemade API for scientific article scrapping

import os
import shutil
import time
import pytz
import datetime
import pandas as pd
from scihub import SciHub
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options


def get_metadata_wos(query: str):
    """Get metadata (title, authors, DOI, etc.) of papers coming from Web of Science

    This function provides metadata of scientific articles coming from the
    results of a searching query on Web of Science (https://www.webofscience.com/wos/woscc/advanced-search).
    The metadata collected for each article are:
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
    query (str): searching query. Sythax (https://webofscience.help.clarivate.com/en-us/Content/wos-core-collection/woscc-search-field-tags.htm).

    Returns:
    The path of pickle file containing metadata.
    """

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

    print(f"Begin scrapping at {utc_current_time_str}.")

    tmp_xls_files = "./data/tmp_" + utc_current_time_str
    os.makedirs(tmp_xls_files)

    url = "https://www.webofscience.com/wos/woscc/advanced-search"
    firefox_options = Options()
    firefox_options.add_argument("-headless")
    driver = webdriver.Firefox(options=firefox_options)
    driver.get(url)

    print(f"Connected to Web of Science.")

    time.sleep(3)
    driver.find_element(By.ID, "onetrust-reject-all-handler").click()

    time.sleep(2)
    search_input = driver.find_element(By.ID, "advancedSearchInputArea")
    search_input.send_keys(query)
    search_input.submit()

    print(f"Searching papers corresponding to query: {query}...")

    time.sleep(2)
    nb_results = driver.find_element(
        By.XPATH,
        "/html/body/app-wos/main/div/div/div[2]/div/div/div[2]/app-input-route/app-base-summary-component/app-search-friendly-display/div[1]/app-general-search-friendly-display/div[1]/h1/span",
    ).text
    nb_results = int(nb_results.replace(",", ""))

    print(f"Found {nb_results} papers.")

    def downloader(input_aera_idex, sent_key_low, sent_key_high):
        time.sleep(2)
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
            By.ID, "mat-input-" + str(2 * input_aera_idex + 1)
        )
        low_bound.clear()
        low_bound.send_keys(str(sent_key_low))
        high_bound = driver.find_element(
            By.ID, "mat-input-" + str(2 * input_aera_idex + 2)
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

        time.sleep(5)
        local_downloads_path = os.path.expanduser("~/Downloads")
        full_path_downloaded_files = []
        for i in local_downloads_path:
            full_path_downloaded_files.append(local_downloads_path + "/" + i)
        latest_file = max(full_path_downloaded_files, key=os.path.getctime)
        shutil.move(
            latest_file, os.path.join(tmp_xls_files, str(batch_nb + 1) + ".xls")
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

    print(f"Cleaning the dataset...")

    data = []
    for filename in os.listdir(tmp_xls_files):
        file_path = os.path.join(tmp_xls_files, filename)
        df = pd.read_excel(file_path)
        data.append(df)
    df = pd.concat(data)

    shutil.rmtree(tmp_xls_files)

    attributes_to_keep = [
        "DOI",
        "Article Title",
        "Authors",
        "Publication Year",
        "Abstract",
        "Author Keywords",
        "Language",
        "Affiliations",
        "Source Title",
        "Publisher",
        "Volume",
        "Issue",
    ]
    size_before = len(df)
    df = df[attributes_to_keep]
    df = df.drop_duplicates()
    df = df[df["DOI"].duplicated(keep=False) == False]
    size_after = len(df)
    size_diff = size_before - size_after
    proportion = round(size_diff * 100 / size_before, 1)

    print(f"{size_diff} duplicates deleted ({proportion}%).")

    print("\nSummary Information:")
    print(df.info())

    data_folder = os.makedirs("./data/" + utc_current_time_str)
    metadata_file = os.path.join(data_folder, "metadata.pkl")
    df.to_pickle(metadata_file)

    print(f"\nMetadata of {size_after} papers saved to {metadata_file}.")

    return metadata_file


def get_metadata_scholar(query: str):
    """Quick description

    Long description

    Parameters:
    query (str):

    Returns:

    """
    pass


def get_metadata_scopus(query: str):
    """Quick description

    Long description

    Parameters:
    query (str):

    Returns:

    """
    pass


def get_pdf_scihub(metadata: str):
    """Quick description

    Long description

    Parameters:
    metadata (str):

    Returns:

    """
    pass


def main():
    get_metadata_wos("ts = (specific searching query)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("An error occurred: {}".format(str(e)))
