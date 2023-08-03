def run(query: str, platforms: list[str]):
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
    import os

    sys.path.append("..")
    import src

    platform_scrappers = {
        "WoS": src.scrapper.wos
    }  # To be completed when new platforms supported

    platform_messages = {
        "WoS": "Web of Science Core Collection"
    }  # To be completed when new platforms supported

    if platforms == []:
        print(
            "Parameter 'platforms' is empty. At least one platform must be specified."
        )
    else:
        valid_plateform_inputs = []
        for platform in platforms:
            if platform in platform_scrappers:
                valid_plateform_inputs.append(platform)
            else:
                print(
                    f"'{platform}' is passed as a platform but not supported by colibri. Correct and re-run the function."
                )
                return 0

        import pytz
        import datetime
        import os

        valid_message = []
        for platform in valid_plateform_inputs:
            valid_message.append(str(platform_messages.get(platform)))

        sentence_template = "You selected platform(s) {} to get the publications from."
        formatted_sentence = sentence_template.format(", ".join(valid_message))
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

        print(f"\nStart scrapping at {utc_current_time_str} UTC.")

        current_dir = os.getcwd()
        target_folder = "colibri"
        while os.path.basename(current_dir) != target_folder:
            current_dir = os.path.dirname(current_dir)
        data_folder = os.path.join(current_dir, "data/scrapped/" + utc_current_time_str)
        os.makedirs(data_folder)

        for platform in valid_plateform_inputs:
            platform_scrappers.get(platform)(data_folder, query)
