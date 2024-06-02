import json
from environment.item import ItemsLoader
from environment.movies.movie import Movie


class MoviesLoader(ItemsLoader):
    """
    The MoviesLoader object is responsable to load movies that are stored in a json file
    """

    def __init__(self, json_file):
        self.dataset_file = json_file
        with open(self.dataset_file) as json_file:
            self.data = json.load(json_file)

    def load_all_ids(self):
        """
        Return a list of all ids in the dataset

        Args:
            None

        Return:
            List of integers containing all ids of all movies in the dataset
        """
        return list(map(lambda x: int(x), self.data.keys()))

    def load_items(self):
        """
        Args:
            None

        Returns:
            movies (dictionary from Movies ids to Movies): A mapping for all movies in the dataset
        """

        movies = {}
        for movie in self.data:
            movies[movie["id"]] = Movie.from_json(movie)
        return movies

    def load_items_from_ids(self, id_list):
        """
        Args:
            id_list (list of integers): the ids of some movies

        Return:
            movies (list of Movies): A list for all movies that correspond to the given indices
        """
        movies = []
        for id in id_list:
            movies.append(Movie.from_json(self.data[str(id)]))
        return movies
