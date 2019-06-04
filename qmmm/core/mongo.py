
from pymongo import MongoClient
from bson import Binary
from .configuration import Configuration
from logging import getLogger
import numpy as np
import pickle

__GLOBAL_MONGO_CONNECTIONS = {}

def get_mongo_client(host=None, port=None, database=None, collection=None):
    host = host or 'localhost'
    port = int(port) or 27017
    global __GLOBAL_MONGO_CONNECTIONS
    if (host, port) not in __GLOBAL_MONGO_CONNECTIONS:
        client = MongoClient(host=host, port=port)
        __GLOBAL_MONGO_CONNECTIONS[(host, port)] = client
    try:
        client = __GLOBAL_MONGO_CONNECTIONS[(host, port)]

    except:
        getLogger(__name__).exception('Failed to establish connection to mongo server mongodb://{}:{}'.format(host, port))
    else:
        getLogger(__name__).info('Established connection to mongo server mongodb://{}:{}'.format(host, port))
    result = [client]
    if database:
        database_ = client[database]
        result.append(database_)
    if collection and database:
        # Only if both are specified
        collection_ = database_[collection]
        collection_.find_one()
        result.append(collection_)

    return tuple(result)

def insert(collection, dictionary):

    assert isinstance(dictionary, dict)
    # Serialize all numpy arrays with pickle for performance reasons
    def _subsitute_numpy(dictionary_):
        if isinstance(dictionary_, dict):
            for key, value in dictionary_.items():
                if isinstance(value, dict):
                    # It is a dictionary again
                    _subsitute_numpy(value)
                elif isinstance(value, list):
                    # Check all values could be dictionaries again
                    dictionary_[key] = [_subsitute_numpy(v) for v in value]
                elif isinstance(value, tuple):
                    dictionary_[key] = tuple([_subsitute_numpy(v) for v in value])
                else:
                    dictionary_[key] = _subsitute_numpy(value)
            # Return the dictionary itself
            return dictionary_

        elif isinstance(dictionary_, np.ndarray):
            # Could be in a loop
            return Binary(pickle.dumps(dictionary_))
        else:
            return dictionary_

    _subsitute_numpy(dictionary)
    if 'id' in dictionary:
        calc_id = dictionary['id']
        document = query(collection, calc_id, raise_error=False)
        if document:
            update = True
        else:
            update = False
    else:
        update = False
    if not update:
        collection.insert_one(dictionary)
        getLogger(__name__).info('Inserted document "{}" to database'.format(dictionary['id']))
    else:
        assert dictionary['id'] == calc_id
        collection.update_one({
            'id': dictionary['id']
        }, update={
            '$set': {k: v for k, v in dictionary.items() if k != '_id'}
        })
        getLogger(__name__).info('Updated document "{}" in database'.format(dictionary['id']))


def query(collection, calc_id, raise_error=True):

    query_document = {
        'id': calc_id
    }
    dictionary = collection.find_one(filter=query_document)
    if not dictionary:
        if raise_error:
            raise RuntimeError('Failed to retrieve object "{}"'.format(calc_id))
        else:
            return dictionary


    def _subsitute_numpy(dictionary_):
        if isinstance(dictionary_, dict):
            for key, value in dictionary_.items():
                if isinstance(value, dict):
                    # It is a dictionary again
                    _subsitute_numpy(value)
                elif isinstance(value, list):
                    # Check all values could be dictionaries again
                    dictionary_[key] = [_subsitute_numpy(v) for v in value]
                elif isinstance(value, tuple):
                    dictionary_[key] = tuple([_subsitute_numpy(v) for v in value])
                else:
                    dictionary_[key] = _subsitute_numpy(value)
            # Return the dictionary itself
            return dictionary_

        elif isinstance(dictionary_, bytes):
            # Could be in a loop
            return pickle.loads(dictionary_)
        else:
            return dictionary_

    _subsitute_numpy(dictionary)
    return dictionary



def make_database_config(db):
    if isinstance(db, dict):
        keys = ['host', 'port', 'database', 'collection']
        for k in keys:
            if k not in db:
                raise ValueError('Missing key "{}" in database config'.format(k))
        # Everything is ok
    elif isinstance(db, bool):
        if db:
            # Try to fetch settings from configuration
            keys = ['host', 'port', 'database', 'collection']
            config = Configuration()
            options = config.get_options('db')
            db_ = {}
            for k in keys:
                if k not in options:
                    raise ValueError(
                        'Missing key "{}" in database config. Please specify it in the settings file'.format(k))
                else:
                    db_[k] = config.get_option('db', k)
            # Everything worked out
            db = db_
    else:
        raise TypeError('db keyword arguement must be either dict or bool')

    return db