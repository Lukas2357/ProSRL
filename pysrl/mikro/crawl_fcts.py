"""Functions for the crawl.py module"""

import io
import os
from os import path

import yaml
from mechanize import Browser, Link

from crawl_classes import LearnPage
from constants import DATA_PATH
from crawl_help_fcts import *


def reach_main_page(log="LenviHunold", pwd="UcCZeBSr!") -> Browser:
    """Create browser instance and navigate to overview page

    Args:
        log (str, optional): The user login. Defaults to "LenviHunold".
        pwd (str, optional): The user password. Defaults to "UcCZeBSr!".

    Returns:
        Browser: The browser instance currently at the overview page

    """
    br = Browser()
    br.set_handle_robots(False)
    url = "https://lenvi.l3hrit.de/online-lernen/kraft-2/"
    br.open(url)
    br.select_form(nr=0)
    br.form['log'] = log
    br.form['pwd'] = pwd
    br.submit()

    br.select_form(nr=3)
    br.form["totalpoll[fields][bitte_geben_sie_hier_" +
            "ihren_pers%c3%b6nlichen_code_ein]"] = "Test"
    br.submit()

    link = br.find_link("Klicken Sie hier zum Starten der Lernumgebung")
    br.follow_link(link)

    link = br.find_link("Überspringen")
    br.follow_link(link)

    return br


def init_learn_page(active_link: Link, link: Link, sublink: Link,
                    subsublink: Link, category: str, topic=None) -> LearnPage:
    """Create a LearnPage object from the given attributes

    Args:
        active_link: The down-most link that is not None
        link (Link): The upmost level link found while crawling here
        sublink (Link): The middle level link found while crawling here
        subsublink (Link): The down-most level link found while crawling here
        category (str): The category of the page extracted before
        topic (str, optional): The topic of the page in case it was extracted 
                                before. Defaults to None.

    Returns:
        LearnPage: A LearnPage object of the given attributes
        
    """

    active_link_corrected = correct_link(active_link)
    label = get_link_label(active_link_corrected)

    if label is None:
        return LearnPage(link.text, None, None, None, None, active_link.url)

    title = None if subsublink is None \
        else get_link_title(active_link_corrected, label)

    if topic is None:
        topic = None if sublink is None else sublink.text
    else:
        title, topic = sublink.text, topic.strip()

    if topic and "Start Version " in topic:
        title, topic = topic.replace("Start", "Test"), None

    return LearnPage(label, title, topic, link.text, category, active_link.url)


def add_learn_page(url: str, pages: list[LearnPage], link: Link,
                   sublink: Union[Link, None], subsublink: Union[Link, None],
                   category: str, verbose: bool, topic=None) -> list[LearnPage]:
    """Add a LearnPage object to the learn-pages list

    Args:
        url (str): The url of the page where the link is found
        pages (list[LearnPage]): The list of LearnPage object to be extended
        link (Link): The upmost level link found while crawling here
        sublink (Link): The middle level link found while crawling here
        subsublink (Link): The down-most level link found while crawling here
        category (str): The category of the page extracted before
        verbose (bool): Whether to log the process
        topic (str, optional): The topic of the page in case it was extracted 
                                before. Defaults to None.

    Returns:
        list[LearnPage]: The extended list of LearnPage objects
        
    """
    if subsublink is not None:
        active_link = subsublink
    else:
        active_link = sublink if sublink is not None else link

    if url in active_link.url and \
            active_link.text != "Starten der gesamten Einheit":

        if verbose:
            print(active_link.text, "---", active_link.url)

        page = init_learn_page(
            active_link, link, sublink, subsublink, category, topic)
        pages.append(page)

        print("Found", len(pages), "learn pages")

    return pages


def get_learn_pages(categories=(0, 6), verbose=0) -> list[LearnPage]:
    """Get all the learn-pages present anywhere on the webpage

    Args:
        categories (tuple, optional): Main categories to parse. Defaults (0, 6).
        verbose (int, optional): Whether info should be logged. Defaults to 0.

    Returns:
        list[LearnPage]: List of LearnPage instances found on the webpage

    """
    br = reach_main_page()
    soup = get_browser_soup(br)
    category_map = get_main_categories(soup)

    pages = []
    category_counter = -1

    for link in br.links():

        if link.text == "Erarbeiten Grundwissen":
            category_counter += 1

        if category_counter < categories[0]:
            continue

        if category_counter > categories[1]:
            return pages

        category = category_map[category_counter]

        page_type = get_page_type(link)

        if "kraft-2/k" in link.url:

            url = br.geturl()
            add_learn_page(url, pages, link, None, None, category, verbose)

            br.follow_link(link)

            for sublink in br.links():

                if "kraft-2/k" in sublink.url:

                    if page_type == "ueben":
                        topic = get_ueben_topic(br, sublink)
                    else:
                        topic = None

                    url = br.geturl()
                    add_learn_page(url, pages, link, sublink, None, category,
                                   verbose, topic)

                    if page_type == "erarbeiten":

                        br.follow_link(sublink)

                        for subsublink in br.links():
                            url = correct_url(br.geturl())
                            add_learn_page(url, pages, link, sublink,
                                           subsublink, category, verbose)

                        br.back()
            br.back()

    return pages


def construct_learn_pages(crawl: list[LearnPage]) -> dict:
    """Construct the learn-pages dict from the crawl resulting

    Args:
        crawl (list[LearnPage]): List of LearnPage instances returned from crawl

    Returns:
        dict: The learn-pages dict

    """
    learn_pages = {}

    for item in crawl:
        args = {"Title": item.title, "Topic": item.topic, "Type": item.type,
                "Category": item.category, "Link": item.link}
        learn_pages[item.label] = args

    return learn_pages


def _additional_learn_pages() -> list[LearnPage]:
    """Create LearnPage objects for learn pages not on the webpage anymore

    Returns:
        list[LearnPage]: A list of defined LearnPage objects

    """

    sub_urls = ["", "ks3a3b1_1a-hangabtriebskraft-einzeichnen/",
                "ks3a3b2_1a-hangabtriebskraft-ueber-gewichtskraft-berechnen/",
                "ks3a3b3_1a-hangabtriebskraft-ueber-gewichtskraft-vergleichen/",
                "ks3a3b4_1a-hangabtriebskraft-zeichnerisch-vergleichen/",
                "ks3a3k4_1a-hangabtriebskraft-vergleichen-ueber-winkel/"]

    urls = ["https://lenvi.l3hrit.de/online-lernen/kraft-2/ks3a-erarbeiten-" +
            "des-grundwissens/ks3a3-hangabtriebskraft/" + sub_url
            for sub_url in sub_urls]

    return [LearnPage("ks3a2",
                      None, "Normalkraft", "Erarbeiten Grundwissen",
                      "Besondere Kräfte", "https://lenvi.l3hrit.de/online-" +
                      "lernen/kraft-2/ks3a-erarbeiten-des-grundwissens/" +
                      "ks3a3-5-normalkraft/"),
            LearnPage("ks3a3", None, "Normalkraft", "Erarbeiten Grundwissen",
                      "Besondere Kräfte", "https://lenvi.l3hrit.de/online-" +
                      "lernen/kraft-2/ks3a-erarbeiten-des-grundwissens/ks3a2-" +
                      "normalkraft/"),
            LearnPage("ks3a3a",
                      None, "Hangabtriebskraft", "Erarbeiten Grundwissen",
                      "Besondere Kräfte", urls[0]),
            LearnPage("ks3a3b1_1a",
                      "hangabtriebskraft einzeichnen",
                      "Hangabtriebskraft", "Erarbeiten Grundwissen",
                      "Besondere Kräfte", urls[1]),
            LearnPage("ks3a3b2_1a",
                      "hangabtriebskraft ueber gewichtskraft berechnen",
                      "Hangabtriebskraft", "Erarbeiten Grundwissen",
                      "Besondere Kräfte", urls[2]),
            LearnPage("ks3a3b3_1a",
                      "hangabtriebskraft ueber gewichtskraft vergleichen",
                      "Hangabtriebskraft", "Erarbeiten Grundwissen",
                      "Besondere Kräfte", urls[3]),
            LearnPage("ks3a3b4_1a",
                      "hangabtriebskraft zeichnerisch vergleichen",
                      "Hangabtriebskraft", "Erarbeiten Grundwissen",
                      "Besondere Kräfte", urls[4]),
            LearnPage("ks3a3k4_1a",
                      "hangabtriebskraft vergleichen über winkel",
                      "Hangabtriebskraft", "Erarbeiten Grundwissen",
                      "Besondere Kräfte", urls[5]),
            LearnPage("kd1c2ue10",
                      "kraft vergleichend einzeichnen",
                      "Grundgesetze der Translation", "Üben Grundwissen",
                      "Kraft und Translationsbewegungen",
                      "https://lenvi.l3hrit.de/online-lernen/kraft-2/kd1c-" +
                      "ueben-des-grundwissens/kd1c2ue10_1-kraft-vergleichend-" +
                      "einzeichnen/"),
            LearnPage("ks2c5ue7",
                      "kraefte zeichnerisch in raumrichtungen zerlegen",
                      "Zerlege von Kräften", "Üben Grundwissen",
                      "Mehrkraftsysteme",
                      "https://lenvi.l3hrit.de/online-lernen/kraft-2/ks2c-" +
                      "ueben-des-grundwissens/ks2c5ue7_1-kraefte-" +
                      "zeichnerisch-in-raumrichtungen-zerlegen/"),
            LearnPage("ks2d2ue4",
                      "kraft in komponentengleichung berechnen",
                      "Komponentengleichung", "Üben Erweitertes Wissen",
                      "Mehrkraftsysteme",
                      "https://lenvi.l3hrit.de/online-lernen/kraft-2/ks2d-" +
                      "ueben-des-erweiterten-wissens/ks2d2ue4_1-kraft-in-" +
                      "komponentengleichung-berechnen/"),
            LearnPage("ks2c1ue8",
                      "angeben welche kraefte addiert werden duerfen",
                      "Zeichnerische Addition von Kräften",
                      "Erarbeiten Grundwissen", "Mehrkraftsysteme",
                      "https://lenvi.l3hrit.de/online-lernen/kraft-2/ks2c-" +
                      "ueben-des-grundwissens/ks2c1ue8_1-angeben-welche-" +
                      "kraefte-addiert-werden-duerfen/")
            ]


def add_learn_pages() -> dict:
    """Add pages to learn page dict that do not appear on the webpage anymore

    Returns:
        dict: A dict of additional learn pages to be added

    """
    learn_pages = {}

    for item in _additional_learn_pages():
        args = {"Title": item.title, "Topic": item.topic, "Type": item.type,
                "Category": item.category, "Link": item.link}
        learn_pages[item.label] = args

    return learn_pages


def save_learn_pages(learn_pages: dict, filename="learn_pages"):
    """Save learn pages dict to yaml file

    Args:
        learn_pages (dict): The dictionary of learn pages to save
        filename (str, optional): The yaml filename. Defaults to "learn_pages".
        
    """
    filename = path.join(DATA_PATH, filename + '.yaml')
    os.makedirs(DATA_PATH, exist_ok=True)
    with io.open(filename, 'w', encoding='utf8') as file:
        yaml.dump(learn_pages, file, default_flow_style=False,
                  allow_unicode=True, sort_keys=False)
