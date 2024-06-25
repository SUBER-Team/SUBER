from dataclasses import dataclass
from typing import List
from environment.item import Item
import pandas as pd

# Define data classes
@dataclass
class NewsArticle(Item):
    '''
    Data structure for a newsArticle
    '''
    news_id: str
    category: str
    subcategory: str
    title: str
    abstract: str
    url: str
    title_entities: str
    abstract_entities: str

   def __init__(
        self,
        news_id: str,
        category: str,
        subcategory: str,
        title: str,
        title_embeddings: List[float],
        abstract: str,
        abstract_embeddings: List[float],
        url: str,
        title_entities: str,
        abstract_entities: str,
        
    ):
        super().__init__(id, title)
        self.news_id = news_id
        self.category = category
        self.subcategory = subcategory
        self.title = title
        self.title_embeddings = title_embeddings
        self.abstract = abstract
        self.abstract_embeddings = abstract_embeddings
        self.url = url
        self.title_entities = title_entities
        self.abstract_entities = abstract_entities

    @staticmethod
    def from_dataframe(df: pd.DataFrame):
        return NewsArticle(
            news_id = df["news_id"],
            category = df["category"],
            subcategory = df["subcategory"],
            title  = df["title"],
            title_embeddings  = eval(df["title_embeddings"]),
            abstract = df["abstract"],
            abstract_embeddings = eval(df["abstract_embeddings"]),
            url = df["url"],
            title_entities = df["title-entities"],
            abstract_entities = df["title-entities"],
        )




@dataclass
class UserBehavior(Item):
    '''
    Data structure for a user behavior
    '''
    impression_id: str
    user_id: str
    time: str
    history: List[str]
    impressions: List[str]