"""Module to crawl ProSRL webpage for learn pages and their metadata"""

from .crawl_fcts import *
from ..organizer.orga_fcts import load_learn_pages


def do_crawl(crawl=True, add=True):
    """Do crawl fct to be called from main or directly

    Args:
        crawl (bool, optional): Whether to crawl. Defaults to True.
        add (bool, optional): Whether to add manual pages. Defaults to True.
        
    """
    print("Crawl main page to get all learning pages.",
          "This process might take up to 5 minutes...")

    # Crawl for pages and add manually set ones, if options are set, else load
    # from previously created files (if they do not exist error will be raised)

    learn_pages_crawled = construct_learn_pages(get_learn_pages()) if crawl \
        else load_learn_pages(file="learn_pages_crawled")
    learn_pages_added = add_learn_pages() if add \
        else load_learn_pages(file="learn_pages_added")
    learn_pages = learn_pages_crawled | learn_pages_added

    # Save the resulting dicts of learn pages as yaml
    save_learn_pages(learn_pages_crawled, filename="learn_pages_crawled")
    save_learn_pages(learn_pages_added, filename="learn_pages_added")
    save_learn_pages(learn_pages, filename="learn_pages")
    
    # Make another file for ordered learnpages (according to lenvi)
    ordered_pages = make_ordered_learnpages()
    save_learn_pages(learn_pages=ordered_pages, filename="learn_pages_sorted")

    print(f"Finished crawl, generated ROOT/data/learn_pages.yaml")
