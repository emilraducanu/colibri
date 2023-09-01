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
