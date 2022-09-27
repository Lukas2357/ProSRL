"""Helper functions for the crawl_fcts.py module"""

from typing import Union
from bs4 import BeautifulSoup
from mechanize import Browser, Link
import re


def get_browser_soup(browser: Browser) -> BeautifulSoup:
    """Get the content of the browser response as Beautiful Soup

    Args:
        browser (Browser): The mechanize-browser object to get response from

    Returns:
        BeautifulSoup: The content of the browser response as Beautiful Soup
        
    """
    return BeautifulSoup(browser.response().read(), 'html.parser')


def get_main_categories(soup: BeautifulSoup) -> list[str]:
    """Get main categories of topics from page

    On some pages, the main categories are listed as html h5 headers, so
    we can extract them easily in correct order, to associate the pages later.

    Args:
        soup (BeautifulSoup): The soup of the page

    Returns:
        list[str]: A list of h5 headers corresponding to the main categories
        
    """
    return [str(item.contents[0]) for item in soup.find_all('h5')]


def get_link_label(link: str) -> Union[str, None]:
    """Get the unique label for the link of a learn page

    The label is found as the largest pattern of letters and digits forming an
    identifying code, this ensures that the label corresponds to the deepest
    identifier and is not representing a super page of this page

    Args:
        link (str): The link as a string url

    Returns:
        str: The best match label, if none is found None
        
    """
    label_regex = r"(?<=/)[a-z][a-z][0-9][a-z][^-]*"
    matches = re.findall(label_regex, link)

    if not matches:
        return None

    return max(matches, key=lambda x: len(x))


def get_link_title(link: str, label: str) -> str:
    """Get the title for the link of a learn page

    Args:
        link (str): The link as string url
        label (str): The label of the page identified previously

    Returns:
        str: The title of that page (via text after label in url)
        
    """
    title = re.sub("-", " ", link[link.find(label) + len(label):-1])[1:]

    return title


def correct_url(url: str) -> str:
    """Correct the url of a learn page for proper comparison

    Urls might change and if we check that a link is a sublink by checking if 
    the super link is part of it, we might run into problems. This can be 
    manually avoided by correcting wrong urls in here.

    Args:
        url (str): The url string to correct

    Returns:
        str: The corrected url string
        
    """
    return url.replace("ks3a7-zerlegung-der-gewichtskraft", "ks3a2-normalkraft")


def correct_link(link: Link) -> str:
    """Correct the link of a learn page for proper categorization

    Urls might change and if we get information of a link (topic, title, ...) 
    we might run into problems. This can be manually corrected here.

    Args:
        link (Link): The link to correct

    Returns:
        str: The corrected link as string
        
    """
    return link.url.replace("-5", "")


def get_page_type(link: Link) -> str:
    """Get the type of learn page for a given link
    
    Args:
        link (Link): The link to analyse for page_type

    Returns:
        str: erarbeiten or ueben or None depending on page type
        
    """
    if "-ueben-des" in link.url:
        page_type = "ueben"
    else:
        page_type = "erarbeiten" if "erarbeiten" in link.url else None

    return page_type


def get_ueben_topic(br: Browser, sublink: Link) -> str:
    """Get the topic of an ueben page for a given link

    On the ueben pages we have headers and some pages below. Here we extract the
    positions of the headers and create a dictionary with keys of all headers 
    and the links beneath them as values. Then we find the sublink in the values
    and return the key as the topic of that sublink.

    Args:
        br (Browser): The current browser object
        sublink (Link): The sublink for which we determine the topic

    Returns:
        str: The topic of the sublink
        
    """
    soup = get_browser_soup(br)

    h5_tags = [str(item.contents[0]) for item in soup.find_all(['h5'])]
    all_tags = [str(item.contents[0]) for item in soup.find_all(['h5', 'a'])]

    topic_dict = {}

    for idx, item in enumerate(h5_tags[:-1]):
        start = all_tags.index(item)
        end = all_tags.index(h5_tags[idx + 1])
        topic_dict[item] = all_tags[start:end]

    topic_dict[h5_tags[-1]] = all_tags[all_tags.index(h5_tags[-1]):]

    topic = None
    for key, value in topic_dict.items():
        if sublink.text in value:
            topic = key
            break

    return topic
