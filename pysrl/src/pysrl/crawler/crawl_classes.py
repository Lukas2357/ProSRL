"""Classes for the crawl.py module"""


class LearnPage:
    """The Learn Page class to store information about each LearnPage
    
    """
    def __init__(self, label, title, topic, typ, category, link):
        self.label = label
        self.title = title
        self.topic = topic
        self.type = typ
        self.category = category
        self.link = link
