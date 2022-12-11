import numpy as np

# load the bert vecs
bert_filenames = ['short_description_bert_outputs.npy',
                  'short_description_bert_outputs_cleaned.npy',
                  'about_the_game_bert_outputs.npy',
                  'about_the_game_bert_outputs_cleaned.npy']

def load_bert_vec(filename):
    """Load the bert vecs from the given file."""
    with open('data/processed/' + filename, 'rb') as f:
        return np.load(f)

def load_all_bert_outputs_combined():
    """Load the bert outputs for all the bert models combined."""
    bert_vecs = [load_bert_vec(filename) for filename in bert_filenames]
    return np.concatenate(bert_vecs, axis=1)

def find_nearest_neighbors(data, query, k=10):
    """Find the k nearest neighbors of the query in the data."""
    # compute the distance between the query and each data point
    distances = np.linalg.norm(data - query, axis=1)

    # sort the distances
    sorted_distances = np.argsort(distances)

    # return the indices of the k nearest neighbors
    return sorted_distances[:k]

def find_nearest_neighbors_using_cosine_similarity(data, query, k=10):
    """Find the k nearest neighbors of the query in the data."""
    # compute the cosine similarity between the query and each data point
    # not normalizing because it doesn't affect the order of the results
    similarities = np.dot(data, query)

    # sort the similarities
    sorted_similarities = np.argsort(similarities)

    # return the indices of the k nearest neighbors
    return sorted_similarities[-k:]