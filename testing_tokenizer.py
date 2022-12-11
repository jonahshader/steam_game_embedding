from transformers import BertTokenizer, BertModel, pipeline
import numpy as np

from data_analysis import load_steam_app_data


data = load_steam_app_data('data/download/steam_app_data.csv')[1]
about_the_game = data['about_the_game']

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
encoded_input = tokenizer(about_the_game, return_tensors='pt', padding=True)
print(encoded_input['input_ids'].shape)
print(about_the_game)

# print number of spaces in about_the_game
print(about_the_game.count(' '))