import os
import numpy as np
import pandas as pd
import csv
from dataclasses import dataclass
from typing import List, Dict
import time
import ast
import wikidata.client

# Define paths to the MIND large dataset
base_path = os.path.expanduser('~/efs/resources/datasets/MIND/')
train_path = os.path.join(base_path, 'MINDlarge_train')
dev_path = os.path.join(base_path, 'MINDlarge_dev')
test_path = os.path.join(base_path, 'MINDlarge_test')

# Paths to the embedding files
entity_train_embedding_path = os.path.join(train_path, 'entity_embedding.vec')
relation_train_embedding_path = os.path.join(train_path, 'relation_embedding.vec')
entity_dev_embedding_path = os.path.join(dev_path, 'entity_embedding.vec')
relation_dev_embedding_path = os.path.join(dev_path, 'relation_embedding.vec')
entity_test_embedding_path = os.path.join(test_path, 'entity_embedding.vec')
relation_test_embedding_path = os.path.join(test_path, 'relation_embedding.vec')


# Define column names for the datasets
news_columns = ['id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
behaviors_columns = ['impression_id', 'user_id', 'time', 'history', 'impressions']


def load_data(file_path, column_names):
    '''
    Function to load data into a Dataframe
    '''
    #print("---file_path is {}".format(file_path))
    return pd.read_csv(file_path, sep='\t', names=column_names)



def load_embeddings(path):
    # Read the embeddings file with pandas
    df = pd.read_csv(path, sep='\t', header=None)
    # Assuming the first column is the WikidataId and the rest are embeddings
    wikidata_ids = df.iloc[:, 0].values
    embeddings = df.iloc[:, 1:].values
    return wikidata_ids, embeddings
    
# Function to get the embedding for a specific entity in the dictionary
def get_entity_embedding(news_article, entity_embeddings):
    entity_id = int(news_article['entity'])  # Assuming 'entity' column holds the entity ID
    return entity_embeddings[entity_id]

# Function to get the nearest neighbors
def search_entity_embedding(query_embedding, index, k=5):
    D, I = index.search(query_embedding, k)
    return D, I


# Function to extract the first entity ID from a column
def extract_first_entity_id(entities_column):
    if entities_column:
        entities_list = ast.literal_eval(entities_column)
        if entities_list:
            return entities_list[0]['WikidataId']
    return None


def print_duration(start_time, end_time):
    # Calculate the duration
    duration = end_time - start_time
    
    # Convert the duration to hours, minutes, and seconds
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = duration % 60
    
    # Print the formatted duration
    print(f"Process duration: {hours} hours, {minutes} minutes, and {seconds:.2f} seconds")


def get_wikidata_item_info(item_id):
    '''
    Function leveraging the wikidata API
    '''
    
    # Initialize the client
    client = wikidata.client.Client()

    # Fetch the item
    item = client.get(item_id, load=True)

    # Prepare the output
    item_info = {
        "Item ID": item.id,
        "Label": item.label,
        "Description": item.description,
        "Statements": []
    }

    # Print stuff
    for prop_id, value in item.data['claims'].items():
        prop_info = {"Property ID": prop_id, "Values": []}
        for statement in value:
            mainsnak = statement['mainsnak']
            if mainsnak['datatype'] == 'wikibase-item':
                entity_id = mainsnak['datavalue']['value']['id']
                entity = client.get(entity_id, load=True)
                prop_info["Values"].append(f"{entity.label} ({entity_id})")
            elif mainsnak['datatype'] == 'string':
                prop_info["Values"].append(mainsnak['datavalue']['value'])
            # Add more datatype handlers as needed
        item_info["Statements"].append(prop_info)

    return item_info

def save_catagories_to_csv(df):

    news_cats_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../../mind/datasets/news_cats.csv",
    )

    unique_news_categories_list = df['category'].unique().tolist()
    unique_news_subcategories_list = df['subcategory'].unique().tolist()
    unique_combined_list = list(set(unique_news_categories_list + unique_news_subcategories_list))

    with open(news_cats_path, 'w', newline='') as file:
        writer = csv.writer(file)
        # If you want each item on a new row
        for item in unique_combined_list:
            writer.writerow([item])
    print("--- catagories save to file")
        # If you want all items in a single row, uncomment the line below and comment the loop above
        # writer.writerow(my_list)