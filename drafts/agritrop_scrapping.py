import requests
import xmltodict

pages = 1
recordsAll = []


def start_agritrop_import(resumptionToken=None):
    global pages, recordsAll
    if not resumptionToken:
        recordsAll = []

    url = "https://agritrop.cirad.fr/cgi/oai2"
    params = {
        "verb": "ListRecords",
        "resumptionToken" if resumptionToken else "metadataPrefix": "oai_dc",
        "set": "CTS_2_2019",
    }

    response = requests.get(url, params=params)
    result = xmltodict.parse(response.text, dict_constructor=dict)

    records = result["OAI-PMH"]["ListRecords"]["record"]
    resumptionToken = result["OAI-PMH"]["ListRecords"].get("resumptionToken")

    recordsAll.extend(records)

    if resumptionToken:
        pages += 1
        print("resum:", pages, resumptionToken)
        start_agritrop_import(resumptionToken.get("_text"))
    else:
        print("recordsAll", len(recordsAll))
        do_agritrop_import(recordsAll)
        print("no resumptionToken")


start_agritrop_import()
