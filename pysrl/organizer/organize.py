"""Organizer for the data before analysis (clustering, classification, ...)"""

from ..config.helper import load_input, save_data
from .orga_fcts import *


def do_orga(formats=('csv', )):
    """Do orga function to be called from main or directly

    Args:
        formats (tuple): The formats to save resulting data to (csv and/or xlsx)

    """
    print("Organize data_complete by mapping with learn_pages.yaml")

    # Load learn_pages overview and data_file:
    learn_pages = load_learn_pages()
    correct_logfile()
    required_columns = ['Link', 'Date/Time', 'User']
    data_df = load_input('data_complete.csv', columns=required_columns)

    # Map the links of the loaded raw_data to the attributes of the learn-pages:
    mapped_links = []
    for link in data_df["Link"]:
        mapped_links.append(map_link(link, learn_pages))

    mapped_links = pd.DataFrame(mapped_links)
    mapped_links.drop(['Link'], inplace=True, axis=1)

    # Concat the raw_data with the pages attributes
    data_df = pd.concat([mapped_links, data_df], axis=1)

    # Save the resulting raw_data frame as csv:
    save_data(data_df, 'data_clean', formats=formats)

    print("Orga finished, generated ROOT/raw_data/data_clean.csv")
