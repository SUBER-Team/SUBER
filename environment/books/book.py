from typing import List
from environment.item import Item
import pandas as pd


class Book(Item):
    id: str
    title: str
    description: str
    authors: List[str]
    publisher: str
    published_year: str
    categories: List[str]
    vote_average: float

    def __init__(
        self,
        id: str,
        title: str,
        description: str,
        description_embedding: List[float],
        authors: List[str],
        publisher: str,
        published_year: str,
        categories: List[str],
        vote_average: float,
    ):
        super().__init__(id, title)
        self.title = title
        self.description = description
        self.description_embedding = description_embedding
        self.authors = authors
        self.publisher = publisher
        self.published_year = published_year
        self.categories = categories
        self.vote_average = vote_average

    @staticmethod
    def from_dataframe(df: pd.DataFrame):
        return Book(
            id=df["book_id"],
            title=df["title"],
            description=df["description"],
            description_embedding=eval(df["description_embedding"]),
            authors=eval(df["authors"]) if type(df["authors"]) == str else [],
            publisher=df["publisher"],
            published_year=df["published_year"],
            categories=(
                eval(df["categories"]) if type(df["categories"]) == str else []
            ),
            vote_average=df["vote_average"],
        )
