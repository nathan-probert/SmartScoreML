import pandas as pd
from constants import FEATURES
from utility import apply_model


def make_predictions(players):
    players_df = pd.DataFrame(players)
    refined_data = players_df[["id", "name", "tims", "team_name"] + FEATURES]

    refined_data = apply_model(refined_data)

    for index, row in refined_data.iterrows():
        players[index]["ml_stat"] = row["probability"]

    return players
