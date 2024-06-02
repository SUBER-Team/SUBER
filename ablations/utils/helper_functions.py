import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import go, make_subplots
import tqdm


def interact_sequential(env, reset=True, tqdm_disabled=True):
    num_users = env.num_users
    num_items = env.num_items
    if reset:
        env.reset(seed=42)

    vote_average_tmdb = np.zeros(shape=(num_items))
    for i in range(num_items):
        vote_average_tmdb[i] = env.items_loader.load_items_from_ids(
            [env.action_to_item[i]]
        )[0].vote_average

    data = []
    for user in tqdm.tqdm(range(num_users), disable=tqdm_disabled):
        if reset:
            env.reset(user_id=user)
        else:
            env._user = env.user_list[user]
            env._items_interact = tuple()

        for item in range(num_items):
            obs, reward, terminated, _, info = env.step(item)

            data.append(
                pd.DataFrame(
                    {
                        "item": item,
                        "user": user,
                        "user_name": env.user_list[user].name,
                        "item_name": env.items_loader.load_items_from_ids(
                            [env.action_to_item[item]]
                        )[0].title,
                        "LLM_rating": info["LLM_rating"],
                        "LLM_interaction_HTML": info["LLM_interaction_HTML"],
                        "rating": reward,
                    },
                    index=[0],
                )
            )
    return pd.concat(data), vote_average_tmdb


def interact_sequential_ids(env, users_ids, items_ids, tqdm_disabled=True):
    num_users = env.num_users
    num_items = env.num_items
    env.reset(seed=42)

    vote_average_tmdb = np.zeros(shape=(num_items))
    for i in items_ids:
        vote_average_tmdb[i] = env.items_loader.load_items_from_ids(
            [env.action_to_item[i]]
        )[0].vote_average

    data = []

    for user in tqdm.tqdm(users_ids, disable=tqdm_disabled):
        for item in items_ids:
            env.reset(user_id=user)
            obs, reward, terminated, _, info = env.step(item)
            data.append(
                pd.DataFrame(
                    {
                        "item": item,
                        "user": user,
                        "user_name": env.user_list[user].name,
                        "item_name": env.items_loader.load_items_from_ids(
                            [env.action_to_item[item]]
                        )[0].title,
                        "LLM_explanation": info["LLM_explanation"],
                        "LLM_rating": info["LLM_rating"],
                        "LLM_interaction_HTML": info["LLM_interaction_HTML"],
                        "rating": reward,
                    },
                    index=[0],
                )
            )
    return pd.concat(data), vote_average_tmdb


def data_to_matrix_ids(env, df, users_ids, items_ids):
    num_users = len(np.unique(users_ids))
    num_items = len(np.unique(items_ids))
    ratings = np.zeros(shape=(num_items, num_users))
    for user in users_ids:
        for item in items_ids:
            ratings[item, user] = df[(df["user"] == user) & (df["item"] == item)][
                "rating"
            ].values[0]
    return ratings


def data_to_matrix(env, df):
    num_users = env.num_users
    num_items = env.num_items
    ratings = np.zeros(shape=(num_items, num_users))
    for user in range(num_users):
        for item in range(num_items):
            ratings[item, user] = df[(df["user"] == user) & (df["item"] == item)][
                "rating"
            ].values[0]
    return ratings


def interact_and_condition_explanation(env, out_of_dist_condition, reset=True):
    num_users = env.num_users
    num_items = env.num_items
    if reset:
        env.reset(seed=42)

    vote_average_tmdb = np.zeros(shape=(num_items))
    for i in range(num_items):
        vote_average_tmdb[i] = env.items_loader.load_items_from_ids(
            [env.action_to_item[i]]
        )[0].vote_average

    data = []
    NUM_EXPLANATION_PER_USER = 2
    NUM_EXPLANATION_OUT_OF_DIST = 5
    for user in range(num_users):
        if reset:
            env.reset(user_id=user)
        else:
            env._user = env.user_list[user]
            env._items_interact = tuple()

        count_explanation = 0
        count_out_of_dist = 0
        for item in range(num_items):
            obs, reward, terminated, _, info = env.step(item)

            if (
                out_of_dist_condition(reward)
                and count_out_of_dist < NUM_EXPLANATION_OUT_OF_DIST
            ) or (
                not out_of_dist_condition(reward)
                and count_explanation < NUM_EXPLANATION_PER_USER
            ):
                # Query 2 explanations per user in distribution
                if out_of_dist_condition(reward):
                    count_out_of_dist += 1
                else:
                    count_explanation += 1

                env.delete_last_user_item(user, item)
                tmp = env.rating_prompt.llm_query_explanation
                env.rating_prompt.llm_query_explanation = True
                obs, reward, terminated, _, info = env.step(item)
                env.rating_prompt.llm_query_explanation = tmp

            data.append(
                pd.DataFrame(
                    {
                        "item": item,
                        "user": user,
                        "user_name": env.user_list[user].name,
                        "item_name": env.items_loader.load_items_from_ids(
                            [env.action_to_item[item]]
                        )[0].title,
                        "LLM_explanation": info["LLM_explanation"],
                        "LLM_rating": info["LLM_rating"],
                        "LLM_interaction_HTML": info["LLM_interaction_HTML"],
                        "rating": reward,
                    },
                    index=[0],
                )
            )
    return pd.concat(data), vote_average_tmdb


def plot_heatmap(matrix, title):
    fig = px.imshow(
        matrix, title=title, zmin=1, zmax=10
    )  # Make the zmin, zmax fixed to compare across plots
    fig.update_xaxes(title="user")
    fig.update_yaxes(title="item")
    return fig


def plot_heatmap_2_sides(matrix1, matrix2, title, subtitle1, subtitle2):
    fig = px.imshow(
        np.array([matrix1, matrix2]), title=title, zmin=1, zmax=10, facet_col=0
    )  # Make the zmin, zmax fixed to compare across plots
    fig.update_xaxes(title="user")
    fig.update_yaxes(title="item")
    for i, label in enumerate([subtitle1, subtitle2]):
        fig.layout.annotations[i]["text"] = label
    return fig


def plot_users(data, title):
    fig = px.scatter(
        x=data.mean(axis=0),
        y=data.std(axis=0),
        title=title,
        marginal_x="histogram",
        marginal_y="histogram",
    )
    fig.update_xaxes(title="mean")
    fig.update_yaxes(title="std")
    return fig


def plot_tmdb_corr(data, tmdb_average, title):
    fig = px.scatter(
        x=data.mean(axis=1),
        y=tmdb_average,
        title=title,
        trendline="ols",
    )
    fig.update_xaxes(title="Film average (our)", range=[1, 10])
    fig.update_yaxes(title="TMDB average", range=[1, 10])
    return fig


def header_report(name, prompt, success):
    table = go.Figure(
        data=[
            go.Table(
                cells=dict(
                    values=[
                        ["Name", "Prompt", "Perc. success"],
                        [name, prompt, success],
                    ]
                ),
                columnwidth=[0.15, 0.85],
            ),
        ]
    )
    return table


def header_report_positive_negative(
    name, prompt, success, success_positive, success_negative
):
    table = go.Figure(
        data=[
            go.Table(
                cells=dict(
                    values=[
                        [
                            "Name",
                            "Prompt",
                            "Perc. success",
                            "Perc. success (positive)",
                            "Perc. success (negative)",
                        ],
                        [name, prompt, success, success_positive, success_negative],
                    ]
                ),
                columnwidth=[0.15, 0.85],
            ),
        ]
    )
    return table


def explanation_ratings_report_observe(df, title):
    mask = df["LLM_explanation"].str.len() > 0
    df = df.loc[mask]
    table = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[
                        "User",
                        "Movie",
                        "Rating",
                        "Name",
                        "Title item",
                        "LLM Explanation",
                    ],
                ),
                cells=dict(
                    values=[
                        df["user"],
                        df["item"],
                        df["rating"],
                        df["user_name"],
                        df["item_name"],
                        df["LLM_explanation"],
                    ],
                ),
                columnwidth=[0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.60],
            )
        ],
    )
    return table


def explanation_ratings_report(df, title, out_of_dist_fn):
    mask = df["LLM_explanation"].str.len() > 0
    df = df.loc[mask]
    colors = df["rating"].apply(
        lambda x: "lightsalmon" if out_of_dist_fn(x) else "palegreen"
    )
    table = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[
                        "User",
                        "Movie",
                        "Rating",
                        "Should be",
                        "Name",
                        "Title item",
                        "LLM Explanation",
                    ],
                ),
                cells=dict(
                    values=[
                        df["user"],
                        df["item"],
                        df["rating"],
                        df["should_be"],
                        df["user_name"],
                        df["item_name"],
                        df["LLM_explanation"],
                    ],
                    line_color=[colors],
                    fill_color=[colors],
                ),
                columnwidth=[0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.60],
            )
        ],
    )
    return table


def html_report(body, path_file):
    # a simple HTML template
    s = ""
    for b in body:
        s += b
        s += "\n"

    template = f"""<html>
    <head>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        {s}
    </body>

    </html>"""

    with open(f"{path_file}.html", "w") as f:
        f.write(template)
