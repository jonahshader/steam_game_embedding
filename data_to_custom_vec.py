import json
import numpy as np

from data_analysis import load_steam_app_data

# i will create my own vector representation of the game using the attributes:
# required_age, is_free, price_overview, categories, genres, release_date

# price overview contains the following keys: currency, initial, final
# non-USD currencies are rare, so i will give them the median USD price for simplicity
# on the first pass they will be given None, but later they will be replaced with the median

def field_to_json(field):
    """Convert a field to a json object. """
    if field == '':
        return None
    else:
        return json.loads(field.replace("\'", "\""))

def discover_enum_values(data, attribute, subattribute):
    """Used for many hot encoding of categorical data."""
    values = set()
    for game in data:
        j = field_to_json(game[attribute])
        if j is not None:
            for item in j:
                values.add(item[subattribute])
    return values

def many_hot_encode(values, all_values):
    vec = []
    for value in all_values:
        if value in values:
            vec.append(1)
        else:
            vec.append(0)
    return vec


def make_vec_from_game(game, all_categories, all_genres):
    vec = []
    # required age
    vec.append(0 if game['required_age'] == '' else int(game['required_age']))
    # combine 'is free' and 'price_overview' into one attribute
    if game['is_free'] == 'True':
        vec.append(0)
    else:
        j = field_to_json(game['price_overview'])
        if j is None:
            vec.append(None)
        elif j.get('currency', None) == 'USD':
            vec.append(j.get('final', None))
        else:
            vec.append(None)

    # categories
    if game['categories'] is None:
        vec.extend([0] * len(all_categories))
    else:
        j = field_to_json(game['categories'])
        if j is None:
            vec.extend([0] * len(all_categories))
        else:
            vec.extend(many_hot_encode([item['description'] for item in j], all_categories))
    # genres
    if game['genres'] is None:
        vec.extend([0] * len(all_genres))
    else:
        j = field_to_json(game['genres'])
        if j is None:
            vec.extend([0] * len(all_genres))
        else:
            vec.extend(many_hot_encode([item['description'] for item in j], all_genres))

    # # release date
    # if game['release_date'] is None:
    #     vec.append(None)
    # else:
    #     j = field_to_json(game['release_date'])
    #     if j is None:
    #         vec.append(None)
    #     else:
    #         vec.append(j.get('date', None))

    return vec


def make_raw_vecs():
    data = load_steam_app_data('data/download/steam_app_data.csv')
    all_categories = discover_enum_values(data, 'categories', 'description')
    all_genres = discover_enum_values(data, 'genres', 'description')
    vecs = []
    for game in data:
        vecs.append(make_vec_from_game(game, all_categories, all_genres))
    # make header
    header = ['required_age', 'price']
    header.extend(all_categories)
    header.extend(all_genres)
    return vecs, header

def normalize_vecs(vec):
    """Given a list of lists, normalize each list to have a mean of 0 and a standard deviation of 1. Replace None with mean"""
    # first convert to numpy array
    vec = np.array(vec, dtype=np.float64)
    # replace None with mean
    # https://stackoverflow.com/questions/18689235/numpy-array-replace-nan-values-with-average-of-columns
    col_mean = np.nanmean(vec, axis=0)
    inds = np.where(np.isnan(vec))
    vec[inds] = np.take(col_mean, inds[1])

    # normalize each column
    mean = np.mean(vec, axis=0)
    std = np.std(vec, axis=0)
    return (vec - mean) / std

def make_vecs():
    vecs, h = make_raw_vecs()
    return normalize_vecs(vecs), h
