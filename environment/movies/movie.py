from environment.item import Item


class Actor:
    """
    Object used to store actors informations
    """

    def __init__(self, gender, id, name, popularity, character):
        """
        gender (string): gender of the actor 'M' for male and 'F' for female
        id (integer): unique identifiers
        name (string): name of the actor
        popularity (integer): score created internally in TMDB that represents how much an actor is popular
        character (string): name of the character that the actor portrays
        """
        self.gender = gender
        self.id = id
        self.name = name
        self.popularity = popularity
        self.character = character


class Movie(Item):
    """
    From TMDB, features:
    id (int): TMDB id
    adult (bool): flag if the film is for adults
    genres (string): list of names and ids of the film's genres
    imdb_id (integers): IMDB id
    original_title (string): title in the original language
    title (string): integer translation of the title
    original_language (string): original language of the film
    overview (string): description of the film
    popularity (integer): popularity score of the film
    release date (string): date of release of the film
    revenue (integer)
    budget (integer)
    vote average (integer): average vote given by users
    vote count (integer): amout of votes recived
    runtime (TODO check type)
    List of most relevant actors
    """

    def __init__(
        self,
        id,
        imdb_id,
        actors,
        adult,
        budget,
        genres,
        original_language,
        original_title,
        overview,
        overview_embedding,
        popularity,
        release_date,
        revenue,
        runtime,
        title,
        vote_average,
        vote_count,
        director="",
    ):
        self.id = id
        self.imdb_id = imdb_id
        self.actors = actors
        self.adult = adult
        self.budget = budget
        self.genres = genres
        self.original_language = original_language
        self.original_title = original_title
        self.overview = overview
        self.overview_embedding = overview_embedding
        self.popularity = popularity
        self.release_date = release_date
        self.revenue = revenue
        self.runtime = runtime
        self.title = title
        self.vote_average = vote_average
        self.vote_count = vote_count
        self.director = director

    @staticmethod
    def from_json(data):
        """
        Returns a Movie object based on the Json data

        Args:
            data (json file): json representation of a movie

        Return:
            Movie object that correspond to the Json representation
        """
        actors = []
        for actor in data["actors"]:
            actors.append(
                Actor(
                    "M" if (actor["gender"] == 2) else "F",
                    actor["id"],
                    actor["name"],
                    actor["popularity"],
                    actor["character"],
                )
            )
        genres = []
        for genre in data["genres"]:
            genres.append(genre["name"].lower())

        return Movie(
            data["id"],
            data["imdb_id"],
            actors,
            data["adult"],
            data["budget"],
            genres,
            data["original_language"],
            data["original_title"],
            data["overview"],
            data["overview_embedding"],
            data["popularity"],
            data["release_date"],
            data["revenue"],
            data["runtime"],
            data["title"],
            data["vote_average"],
            data["vote_count"],
            data["director"] if "director" in data else "",
        )
