from pymongo import MongoClient
# pprint library is used to make the output look more pretty
from pprint import pprint

# connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string
client = MongoClient('mongodb://Roomsystem:admin01@202.28.34.197:27017/Roomsystem')


def get_database_by_name(name):
    return client[name]
