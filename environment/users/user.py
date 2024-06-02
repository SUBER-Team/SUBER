class User:
    """
    Object to represent users that will interacts with the s
    """

    id_counter = 0

    def __init__(self, name, gender, age, description, job="", hobby=""):
        """
        id (integer): unique identifier for each user
        name (string): name + surname of the user
        gender (string): 'M' for male and 'F' for female
        age (integer): the age of the user
        description (string): small description of the user, including its cinematic interests
        job (string, optional): the job of the user
        hobby (string, optional): the hobby of the user
        """
        self.id = User.id_counter
        User.id_counter += 1
        self.name = name
        self.gender = gender
        self.age = age
        self.description = description
        self.job = job
        self.hobby = hobby

    def __str__(self) -> str:
        return (
            f"User(id = {self.id}, name = {self.name}, gender = {self.gender}, age ="
            f" {self.age}:\n{self.description}), job = {self.job}, hobby ="
            f" {self.hobby})"
        )

    def __repr__(self) -> str:
        return f"User(id = {self.id}, name = {self.name})"

    @staticmethod
    def get_num_users():
        return User.id_counter
