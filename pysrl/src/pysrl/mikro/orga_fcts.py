"""Functions for the organizer.py module"""

import yaml
from os import path
import pandas as pd

from constants import DATA_PATH


def load_learn_pages(file='learn_pages') -> dict:
    """Load the learn-pages dictionary from the yaml file created by crawl.py

    Args:
        file (string): The yaml file to be loaded, defaults to 'learn_pages'

    Returns:
        dict: The-learn pages dictionary with labels and attributes of pages

    """
    learn_pages_file = path.join(DATA_PATH, file + '.yaml')

    with open(learn_pages_file, 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    return data_loaded


def correct_link_label(link_label: str) -> str:
    """To match the learn-pages dictionary, link labels might need modification

    Args:
        link_label (str): The link_label not matching learn pages dictionary

    Returns:
        str: A modified label found in the dictionary
    """
    if link_label == "ks3a2":
        return "ks3a7"

    return link_label


def get_null_dict(dictionary: dict) -> dict:
    """For a given dict, return the same with all values None

    Args:
        dictionary (dict): The input dict

    Returns:
        dict: The same dict with all values None

    """
    new_dict = {}
    for key in dictionary.keys():
        new_dict[key] = None

    return new_dict


def map_link(link: str, learn_pages: dict) -> dict:
    """Map the link from the data to the learn-pages dictionary

    Args:
        link (str): The link to be mapped
        learn_pages (dict): The dictionary of pages labels and attributes

    Returns:
        dict: The dict with label and attributes of pages

    """

    null_dict = get_null_dict(list(learn_pages.values())[0])

    if pd.isna(link):
        return {"Label": ""} | null_dict

    if link == "https://lenvi.l3hrit.de/":
        return {"Label": "Startseite"} | null_dict

    if link == "https://lenvi.l3hrit.de/einfuehrungs-tour/":
        return {"Label": "Tour"} | null_dict

    if link == "https://lenvi.l3hrit.de/online-lernen/kraft-2/":
        return {"Label": "Übersicht"} | null_dict

    if link == "https://lenvi.l3hrit.de/logout-page/":
        return {"Label": "Logout"} | null_dict

    if link == "https://lenvi.l3hrit.de/impressum/":
        return {"Label": "Impressum"} | null_dict

    if link == "https://lenvi.l3hrit.de/eigene-notizen/":
        return {"Label": "Notizen"} | null_dict

    if "rueckmeldungen-zum-test" in link:
        return {"Label": "Rückmeldungen"} | null_dict

    link_labels = [label for label in learn_pages.keys()
                   if learn_pages[label]["Link"] == link]

    if link_labels:
        return {"Label": link_labels[0]} | learn_pages[link_labels[0]]

    print(f"Link {link} could not be mapped!")
    return {"Label": ""} | null_dict
