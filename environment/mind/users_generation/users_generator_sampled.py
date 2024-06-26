import os
import csv
import guidance
import pandas as pd
import numpy as np
import torch
from environment.LLM.guidance import get_model
import tqdm
import scipy


class UserGeneratorHardLLMFeatures:
    def __init__(self, llm):
        self.llm = llm
        hobbies_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../../users/datasets/hobby_list.csv",
        )
        self.hobbies = pd.read_csv(hobbies_path)

        jobs_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../../users/datasets/job_list.csv",
        )
        self.jobs = pd.read_csv(jobs_path)

        hobbies_children_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../../users/datasets/hobby_children.csv",
        )

        self.hobbies_children = pd.read_csv(hobbies_children_path)

        self.dist = pd.DataFrame(
            {
                "Genre": [
                    "comedy",
                    "adventure",
                    "action",
                    "thriller",
                    "drama",
                    "crime",
                    "documentary",
                    "animated",
                    "fantasy",
                    "science fiction",
                    "romance",
                    "horror",
                ],
                "Liked": [
                    0.9,
                    0.9,
                    0.88,
                    0.84,
                    0.84,
                    0.82,
                    0.77,
                    0.7,
                    0.7,
                    0.69,
                    0.67,
                    0.52,
                ],
            },
        )

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

    def generate_age(self, num):
        age = self.rng.choice(
            a=[4, 15, 25, 35, 45, 55, 65],
            size=num,
            p=[0.182, 0.13, 0.137, 0.131, 0.123, 0.129, 0.168],
        ) + self.rng.integers(low=0, high=10, size=num)
        return age

    def generate_hobby(self, num):
        return self.hobbies.sample(n=num, replace=True, random_state=self.rng)[
            "Hobby-name"
        ].values

    def generate_job(self, num):
        return self.jobs.sample(n=num, replace=True, random_state=self.rng)[
            "Jobs"
        ].values

    def generate_hobby_children(self, num):
        return self.hobbies_children.sample(n=num, replace=True, random_state=self.rng)[
            "Hobby-name"
        ].values

    # def generate_liked_genres(self):
    #     return self.rng.choice(
    #         self.dist["Genre"].values,
    #         size=self.rng.choice(a=[1, 2], p=[0.2, 0.8], size=1),
    #         p=scipy.special.softmax(self.dist["Liked"].values),
    #     )

    # def generate_not_like_genres(self, liked):
    #     remain = self.dist[self.dist["Genre"].apply(lambda x: not x in liked)]
    #     return self.rng.choice(
    #         remain["Genre"].values,
    #         size=self.rng.choice(a=[1, 2], p=[0.8, 0.2], size=1),
    #         p=scipy.special.softmax(remain["Liked"].values),
    #     )

    def generate_genres(self):
        liked = []
        not_liked = []
        for _, g in self.dist.iterrows():
            if self.rng.choice([True, False], p=[g["Liked"], 1.0 - g["Liked"]], size=1):
                liked.append(g["Genre"])
            else:
                not_liked.append(g["Genre"])

        liked = (
            self.rng.choice(liked, size=min(4, len(liked)), replace=False)
            if len(liked) > 0
            else []
        )
        not_liked = (
            self.rng.choice(not_liked, size=min(4, len(not_liked)), replace=False)
            if len(not_liked) > 0
            else []
        )
        return liked, not_liked

    def generate_normal(self, seed, num):
        ages = self.generate_age(num)
        hobbies = self.generate_hobby(num)
        jobs = self.generate_job(num)
        hobbies_children = self.generate_hobby_children(num)

        genres_liked = []
        genres_disliked = []
        genres_liked_str = []
        genres_disliked_str = []
        for i in range(len(ages)):
            if ages[i] > 60:
                jobs[i] = "retired"
            elif ages[i] < 16:
                jobs[i] = "schoolchild"
                hobbies[i] = hobbies_children[i]
            elif ages[i] >= 16 and ages[i] <= 24:
                jobs[i] = self.rng.choice(a=["student", "apprentice"], p=[0.5, 0.5])

            a, b = self.generate_genres()
            genres_liked.append(a)
            genres_disliked.append(b)

            genres_liked_str.append(", ".join(genres_liked[i]))
            genres_disliked_str.append(", ".join(genres_disliked[i]))

        prompt = guidance(
            (
                "{{#user~}}Can you generate different persons, for each of them you"
                " need to generate a name, an age, an hobby, a job and a detailled,"
                " long and original description that contains the person interests and"
                " secondary hobbies."
                + " Please outline the cinematic preferences of the individual,"
                " detailing their favorite and least favorite genres. Kindly provide"
                " explanations for each genre preference."
                + " The generated persons should not be too much similar to each"
                " other.{{~/user}}\n"
                + "{{#assistant~}}"
                + "{{#geneach 'items' num_iterations=num join='\n'}}\n"
                + "Name: {{gen 'this.name' temperature=0.7 stop='\n'}}\n"
                + "Age: {{ages[@index]}}\n"
                + "Gender: {{gen 'this.gender' pattern='M|F' temperature=0}}\n"
                + "Hobby: {{hobbies[@index]}}\n"
                + "Job: {{jobs[@index]}}\n"
                + "Genres liked: {{genres_liked[@index]}}\n"
                + "Genres disliked: {{genres_disliked[@index]}}\n"
                + "Description: {{this.name}} is a {{ages[@index]}} years old {{#if"
                " this.gender == 'F'}}female{{else}}male{{/if}}, {{#if"
                " this.gender == 'F'}}she{{else}}he{{/if}} is "
                + "{{gen 'this.description' temperature=0.7 max_tokens=300"
                " stop='\n'}}\n"
                + "{{/geneach}}"
                + "{{~/assistant}}"
            ),
            llm=self.llm,
        )

        out = prompt(
            cache_seed=seed,
            ages=ages,
            jobs=jobs,
            hobbies=hobbies,
            num=num,
            caching=False,
            genres_liked=genres_liked_str,
            genres_disliked=genres_disliked_str,
        )

        df = pd.DataFrame(out["items"])
        df["age"] = ages
        df["job"] = jobs
        df["hobby"] = hobbies
        df["genres_liked"] = genres_liked
        df["genres_disliked"] = genres_disliked
        return df

    def generate_user_film_interests(self, seed, num):
        ages = self.generate_age(num)
        hobbies = self.generate_hobby(num)
        jobs = self.generate_job(num)
        hobbies_children = self.generate_hobby_children(num)

        genres_liked = []
        genres_disliked = []
        genres_liked_str = []
        genres_disliked_str = []
        for i in range(len(ages)):
            if ages[i] > 60:
                jobs[i] = "retired"
            elif ages[i] < 16:
                jobs[i] = "schoolchild"
                hobbies[i] = hobbies_children[i]
            elif ages[i] >= 16 and ages[i] <= 24:
                jobs[i] = self.rng.choice(a=["student", "apprentice"], p=[0.5, 0.5])

            a, b = self.generate_genres()
            genres_liked.append(a)
            genres_disliked.append(b)

            genres_liked_str.append(", ".join(genres_liked[i]))
            genres_disliked_str.append(", ".join(genres_disliked[i]))

        prompt = guidance(
            (
                "{{#user~}}Can you generate different persons, for each of them you"
                " need to generate a name, an age, an hobby, a job and a detailled,"
                " long and original description that contains the person interests,"
                " secondary hobbies."
                + " Please outline the cinematic preferences of the individual,"
                " detailing their favorite and least favorite genres. Kindly provide"
                " explanations for each genre preference and a few movies examples"
                " that this person likes a lot."
                + " Furthermore, delve into their specific movie"
                " interests with elaboration. The generated persons should not be too"
                " much similar to each other.{{~/user}}\n"
                + "{{#assistant~}}"
                + "{{#geneach 'items' num_iterations=num join='\n'}}\n"
                + "Name: {{gen 'this.name' temperature=0.7 stop='\n'}}\n"
                + "Age: {{ages[@index]}}\n"
                + "Gender: {{gen 'this.gender' pattern='M|F' temperature=0}}\n"
                + "Hobby: {{hobbies[@index]}}\n"
                + "Job: {{jobs[@index]}}\n"
                + "Genres liked: {{genres_liked[@index]}}\n"
                + "Genres disliked: {{genres_disliked[@index]}}\n"
                + "Description: {{this.name}} is a {{ages[@index]}} years old {{#if"
                " this.gender == 'F'}}female{{else}}male{{/if}}, {{#if"
                " this.gender == 'F'}}she{{else}}he{{/if}} is "
                + "{{gen 'this.description' temperature=0.7 max_tokens=300"
                " stop='\n'}}\n"
                + "{{/geneach}}"
                + "{{~/assistant}}"
            ),
            llm=self.llm,
        )
        out = prompt(
            cache_seed=seed,
            ages=ages,
            jobs=jobs,
            hobbies=hobbies,
            num=num,
            caching=False,
            genres_liked=genres_liked_str,
            genres_disliked=genres_disliked_str,
        )

        df = pd.DataFrame(out["items"])
        df["age"] = ages
        df["job"] = jobs
        df["hobby"] = hobbies
        df["genres_liked"] = genres_liked
        df["genres_disliked"] = genres_disliked
        return df

    def generate_user_dataset(self, p, num, dir, file_name, seed=0):
        self.seed(seed)
        NUM = 4
        assert num % NUM == 0
        base_path = os.path.join(
            dir,
            file_name,
        )
        if os.path.exists(base_path):
            os.remove(base_path)
        for i in tqdm.tqdm(range(num // NUM)):
            c = self.rng.choice([True, False], p=[p, 1 - p])
            if c:
                self.generate_normal(seed, NUM).to_csv(
                    base_path,
                    index=False,
                    quoting=csv.QUOTE_ALL,
                    mode="a",
                    header=not os.path.exists(base_path),
                )
            else:
                self.generate_user_film_interests(seed, NUM).to_csv(
                    base_path,
                    index=False,
                    quoting=csv.QUOTE_ALL,
                    mode="a",
                    header=not os.path.exists(base_path),
                )
            seed += NUM


import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description="Users generator")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num", type=int, default=4)
    parser.add_argument("--split-without-film-interest", type=float, default=0.5)
    parser.add_argument("--file-name", type=str)
    parser.add_argument(
        "--dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "datasets"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    model = get_model("vicuna13B_GPTQ")
    generator = UserGeneratorHardLLMFeatures(model)
    generator.generate_user_dataset(
        args.split_without_film_interest,
        args.num,
        args.dir,
        args.file_name,
        seed=args.seed,
    )
