import os
import json
from environment.item import ItemsLoader
from environment.mind.news import News
from environment.mind.datasets.load_mind_data import load_data, news_columns,train_path, save_catagories_to_csv




class NewsLoader(ItemsLoader):
    '''
    Responsible for loading news articles
    '''

    def __init__(self):
        '''
        Similar to movies we want to pass in data.  Movies does it with a Json file
        We can do it with a dataframe to make it more efficient. IMHO.

        do something different if it's necessary but this is easiest IMHO

        '''
        self.data = load_data(os.path.join(train_path, 'news.tsv'),news_columns)
        self.data.index = self.data["news_id"]
        save_catagories_to_csv(self.data)


    def load_all_ids(self):
        '''
        define this abstract method
        Return a list of integers containing the ids of all movies
        '''
        return self.data.index.tolist()

    def load_items(self):
        '''
        Define the abstract method like books/movies
        returns:
            news (dictionary from news ids to news): a mapping for all 
        '''
        news = {}
        for article in self.data:
            news[article.index] = News.from_dataframe(article)
        return news
    
    def load_items_from_ids(self,id_list):
        '''
        Define the abstract method
        Returns a list of news articles
        '''
        news = []
        for id in id_list:
            print("ID is {} and type is ".format(id,type(id)))
            print("\t{}\t\n".format(self.data.loc[id]))
            news.append(News.from_dataframe(self.data.loc[id]))
        return news

        
