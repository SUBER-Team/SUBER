from dataclasses import dataclass, field
from typing import List, Optional
from environment.item import Item
import pandas as pd

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/sentence-t5-base')

# Define data classes
@dataclass
class News(Item):
    '''
    Data structure for a newsArticle.  Decided to call it news cause that's shorter.
    '''
    id: str
    category: str
    subcategory: str
    title: str
    abstract: str
    url: str
    title_entities: str
    abstract_entities: str
    _title_embeddings: Optional[List[float]] = field(default=None, repr=False, init=False)
    _abstract_embeddings: Optional[List[float]] = field(default=None, repr=False, init=False)
    clicks: float
    impressions: float
    click_through_rate: float
    read_frequency: float
    vote_count: float
    vote_average: float



    def __init__(
        self,
        id: str,
        category: str,
        subcategory: str,
        title: str,
        abstract: str,
        url: str,
        title_entities: str,
        abstract_entities: str,
        clicks: float,
        impressions: float,
        click_through_rate: float,
        read_frequency: float,
        vote_count: float,
        vote_average: float
        
    ):
        super().__init__(id, title)
        self.id = id
        self.category = category
        self.subcategory = subcategory
        self.title = title
        self.abstract = abstract
        self.url = url
        self.title_entities = title_entities
        self.abstract_entities = abstract_entities
        self.clicks = clicks
        self.impression = impressions
        self.click_through_rate = click_through_rate
        self.read_frequency = read_frequency
        self.vote_count = vote_count
        self.vote_average = vote_average

    @property
    def title_embeddings(self):
        if self._title_embeddings is None:
            self._title_embeddings = model.encode(self.title).tolist()
        return self._title_embeddings

    @property
    def abstract_embeddings(self):
        if self._abstract_embeddings is None:
            self._abstract_embeddings = model.encode(self.abstract).tolist()
        return self._abstract_embeddings

    @staticmethod
    def from_dataframe(df: pd.DataFrame):
        return News(
            id = df["id"],
            category = df["category"],
            subcategory = df["subcategory"],
            title  = df["title"],
            abstract = df["abstract"],
            url = df["url"],
            title_entities = df["title_entities"],
            abstract_entities = df["abstract_entities"],
            clicks = df["clicks"],
            impressions = df["impressions"],
            click_through_rate = df["click_through_rate"],
            read_frequency = df["read_frequency"],
            vote_count = df["vote_count"],
            vote_average = df["vote_average"]
        )



# Note that this may not be needed
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


    def __init__(
        self,
        impression_id: str,
        user_id: str,
        time: str,
        history: list[str],
        impressions: List[str],        
    ):
        super().__init__(id, title)
        self.impression_id = impression_id
        self.user_id = user_ids
        self.time = time
        self.history = history
        self.impressions = impressions

    @staticmethod
    def from_dataframe(df: pd.DataFrame):
        return NewsArticle(
            impression_id = df["impression_id"],
            user_id = df["user_id"],
            time = df["time"],
            history  = df["history"],
            impressions = df["impressions"],
        )

