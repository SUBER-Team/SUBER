from .book import Book
import pandas as pd
from environment.item import ItemsLoader


class BooksLoader(ItemsLoader):
    """
    The BooksLoader object is responsable to load books that are stored in a csv file
    """

    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.data.index = self.data["book_id"]

    def load_all_ids(self):
        """
        Return a list of all ids in the dataset

        Args:
            None

        Return:
            List of integers containing all ids of all books in the dataset
        """
        return self.data.index.tolist()

    def load_items(self):
        """
        Args:
            None

        Returns:
            books (dictionary from books ids to books): A mapping for all books in the dataset
        """

        books = {}
        for movie in self.data:
            books[movie.index] = Book.from_dataframe(movie)
        return books

    def load_items_from_ids(self, id_list):
        """
        Args:
            id_list (list of integers): the ids of some books

        Return:
            books (list of books): A list for all books that correspond to the given indices
        """
        books = []
        for id in id_list:
            books.append(Book.from_dataframe(self.data.loc[int(id)]))
        return books
