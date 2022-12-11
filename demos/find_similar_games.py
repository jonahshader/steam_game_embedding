import numpy as np

import game_search
import systems
from data_analysis import load_steam_app_data
from game_search import game_name_prompt
from systems import find_nearest_neighbors, find_nearest_neighbors_using_cosine_similarity
from sklearn.linear_model import LinearRegression

game_embeddings = systems.load_all_bert_outputs_combined()


def find_similar_games(using_cosine_similarity=False):
    """Find games that are similar to a given game."""
    # prompt the user to select a game
    game, game_index = game_name_prompt()

    # find the k nearest neighbors
    k = 10
    if using_cosine_similarity:
        neighbors, neighbor_dists = find_nearest_neighbors_using_cosine_similarity(game_embeddings, game_embeddings[game_index], k=k)
    else:
        neighbors, neighbor_dists = find_nearest_neighbors(game_embeddings, game_embeddings[game_index], k=k)

    # display the results
    print(f'Games similar to {game["name"]}:')
    for i in range(1, k):
        print(f'{i}: {game_search.data[neighbors[i]]["name"], neighbor_dists[i]}')

def game_to_vec(name=None):
    """Find the vector representation of a game."""
    game, game_index = game_name_prompt(name)
    return game_embeddings[game_index]

def find_similar_games_to_vec(vec, using_cosine_similarity=False):
    """Find games that are similar to a given game."""
    # find the k nearest neighbors
    k = 10
    if using_cosine_similarity:
        neighbors, neighbor_dists = find_nearest_neighbors_using_cosine_similarity(game_embeddings, vec, k=k)
    else:
        neighbors, neighbor_dists = find_nearest_neighbors(game_embeddings, vec, k=k)

    # display the results
    print(f'Similar games:')
    for i in range(0, k):
        print(f'{i}: {game_search.data[neighbors[i]]["name"], neighbor_dists[i]}')

def make_axis_model(a_game_vecs, b_game_vecs):
    """Find the axis that separates the two sets of games."""
    a_game_vecs = np.array(a_game_vecs)
    b_game_vecs = np.array(b_game_vecs)

    # train a linear model where a_game_vecs have the label -1 and b_game_vecs have the label 1
    X = np.concatenate([a_game_vecs, b_game_vecs])
    y = np.concatenate([-np.ones(len(a_game_vecs)), np.ones(len(b_game_vecs))])

    # perform linear regression
    model = LinearRegression()
    model.fit(X, y)

    # return the weights of the model
    return model.coef_, model.intercept_

def sort_games_using_axis(w, b, game_vecs):
    """Sort the games using the given axis."""
    scores = game_vecs @ w + b
    return np.argsort(scores), scores

def print_first_and_last_n_games(indices, n=5):
    """Print the first and last n games."""
    print('First n games:')
    for i in range(n):
        print(f'{i}: {game_search.data[indices[i]]["name"]}')
    print('Last n games:')
    for i in range(n):
        print(f'{i}: {game_search.data[indices[-i - 1]]["name"]}')

def evaluate_existing_axis(game_vecs, data, is_group1_lambda):
    """Create two groups based on applying is_group1_lambda to each game in data.
    Then, find the axis that separates the two groups and evaluate it."""
    # create two groups
    group1 = [i for i in range(len(data)) if is_group1_lambda(data[i])]
    group2 = [i for i in range(len(data)) if not is_group1_lambda(data[i])]

    # truncate the larger group to the size of the smaller group
    if len(group1) > len(group2):
        group1 = group1[:len(group2)]
    else:
        group2 = group2[:len(group1)]

    # split into training and testing sets
    train_size = int(len(group1) * 0.8)
    print(f'Training size: {train_size}')
    train_group1 = group1[:train_size]
    train_group2 = group2[:train_size]
    test_group1 = group1[train_size:]
    test_group2 = group2[train_size:]

    # find the axis that separates the two groups
    w, b = make_axis_model(game_vecs[train_group1], game_vecs[train_group2])

    # evaluate the axis
    # if embedding @ w + b > 0, then the game is in group 2
    # if embedding @ w + b < 0, then the game is in group 1

    # evaluate on the training set
    correct = 0
    for i in train_group1:
        if game_vecs[i] @ w + b < 0:
            correct += 1
    for i in train_group2:
        if game_vecs[i] @ w + b > 0:
            correct += 1
    print(f'Training accuracy: {correct / (len(train_group1) + len(train_group2))}')

    # evaluate on the testing set
    correct = 0
    for i in test_group1:
        if game_vecs[i] @ w + b < 0:
            correct += 1
    for i in test_group2:
        if game_vecs[i] @ w + b > 0:
            correct += 1
    print(f'Testing accuracy: {correct / (len(test_group1) + len(test_group2))}')


factorio = game_to_vec('Factorio')
satisfactory = game_to_vec('Satisfactory')
dyson_sphere = game_to_vec('Dyson Sphere Program')
space_engineers = game_to_vec('Space Engineers')
ror1 = game_to_vec('Risk of Rain')
ror2 = game_to_vec('Risk of Rain 2')
tf2 = game_to_vec('Team Fortress 2')
cod = game_to_vec('Call of Duty®: Black Ops')

