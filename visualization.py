from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

import systems
from data_analysis import load_steam_app_data

# visualize game embeddings using PCA
game_embeddings = systems.load_all_bert_outputs_combined()
pca = PCA(n_components=2)
pca.fit(game_embeddings)
game_embeddings_2d = pca.transform(game_embeddings)

# plot the results
# color based on the game's genre
# get the genres
data = load_steam_app_data('data/download/steam_app_data.csv')
is_action = [game['genres'].lower().find('action') != -1 for game in data]
# color is_action red, not is_action blue
colors = ['red' if is_action[i] else 'blue' for i in range(len(is_action))]
plt.scatter(game_embeddings_2d[:, 0], game_embeddings_2d[:, 1], c=colors, s=1.5)
plt.title('Game Embeddings (PCA)')
plt.legend(['Action', 'Not Action'])
plt.show()

# visualize game embeddings using t-SNE
game_embeddings = systems.load_all_bert_outputs_combined()
tsne = TSNE(n_components=2)
game_embeddings_2d = tsne.fit_transform(game_embeddings)

# plot the results

plt.scatter(game_embeddings_2d[:, 0], game_embeddings_2d[:, 1], c=colors, s=1.5)
plt.title('Game Embeddings (t-SNE)')
plt.legend(['Action', 'Not Action'])
plt.show()

