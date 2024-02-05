def is_valid_pdf(pdf_path):
    from fitz import open as open_pdf

    try:
        doc = open_pdf(pdf_path)
        text = ""
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text += page.get_text()
        return len(text.strip()) > 0
    except Exception as e:
        return False


def ask_unpaywall(doi: str, pdf_name: str, pdf_dir: str):
    from subprocess import run
    from unpywall.utils import UnpywallCredentials

    UnpywallCredentials("emil.raducanu@cirad.fr")
    command = f"unpywall download {doi} --filename={pdf_name}.pdf --path={pdf_dir} --progress=True"
    run(command, shell=True, check=True)


def get_pdf(doi_list):
    from os import getcwd
    from os import remove as remove_file
    from os.path import basename, dirname
    from os.path import join as join_path
    from os.path import exists as does_file_exist
    from random import choice
    from string import ascii_lowercase, digits
    from pandas import DataFrame, read_pickle
    from pandas import concat as df_concat
    from tabulate import tabulate

    current_dir = getcwd()
    project_folder = "colibri"
    while basename(current_dir) != project_folder:
        current_dir = dirname(current_dir)
    pdf_dir = join_path(current_dir, "data/pub_pdf")
    pdf_mapping_file = join_path(pdf_dir, "pdf_mapping.pkl")

    if not does_file_exist(pdf_mapping_file):
        DataFrame({"DOI": [], "PDF path": []}).to_pickle(pdf_mapping_file)
    pdf_mapping = read_pickle(join_path(current_dir, "data/pub_pdf/pdf_mapping.pkl"))

    for doi in doi_list:
        if doi not in list(pdf_mapping["DOI"]):
            pdf_name = "".join(choice(ascii_lowercase + digits) for _ in range(6))
            pdf_mapping = df_concat(
                [
                    pdf_mapping,
                    DataFrame(
                        {"DOI": [doi], "PDF path": [pdf_dir + "/" + pdf_name + ".pdf"]}
                    ),
                ],
                ignore_index=True,
            )
            try:
                ask_unpaywall(doi, pdf_name, pdf_dir)

            except Exception:
                continue
    for pdf in pdf_mapping["PDF path"]:
        if not does_file_exist(pdf):
            pdf_mapping.loc[pdf_mapping["PDF path"] == pdf, "PDF path"] = "unreached"
        elif not is_valid_pdf(pdf):
            pdf_mapping.loc[pdf_mapping["PDF path"] == pdf, "PDF path"] = "unreached"
            remove_file(pdf)
    pdfs_found = (
        pdf_mapping["PDF path"].apply(lambda x: 1 if x != "unreached" else 0).sum()
    )
    total_publications = len(pdf_mapping)
    percentage_found = (pdfs_found / total_publications) * 100
    print(
        f"{percentage_found:.2f}% PDFs found, being {pdfs_found} publications over {total_publications}. Here is the first top 5:"
    )
    print(
        tabulate(
            pdf_mapping.head(5), headers=pdf_mapping.columns, tablefmt="heavy_outline"
        )
    )
    pdf_mapping.to_pickle(join_path(current_dir, "data/pub_pdf/pdf_mapping.pkl"))
    return pdf_mapping


def df2json(df, output_file_path: str):
    """Convert dataframe into cilibri final JSON output.

    Utility that convert the output dataframe of the different stage of the pipeline into a JSON file (cf. 'colibri/data/template_output_database')

    Parameters:
    df: Pandas Dataframe that must be converted. Name of columns must respect the output template.
    output_file_path: absolute path of the output JSON file.

    Returns:
    None. Only creates the file.
    """
    import pandas as pd
    import json

    json_data = {"Meta-analyses": df.to_dict(orient="records")}

    with open(output_file_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)
