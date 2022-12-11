from transformers import BertTokenizer, BertModel, pipeline
import numpy as np
from bs4 import BeautifulSoup

from data_analysis import load_steam_app_data


def clean_text(text):
    """Clean text by removing HTML tags and newlines."""
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text().replace('\n', ' ')

device = 'cuda'

print('Loading data...')
data = load_steam_app_data('data/download/steam_app_data.csv')
print('Cleaning data...')
short_description = [clean_text(r['short_description']) for r in data]
print(short_description[0])
print(short_description[100])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print('Loading model...')
model = BertModel.from_pretrained("bert-base-uncased").to(device)
# encoded_input = tokenizer(short_description, return_tensors='pt', padding=True).to('cuda')
#
# output = model(**encoded_input)
# print(output)

# split the data into batches
outputs = []
i = 0
shapes = []
batch_size = 12
for i in range(0, len(short_description), batch_size):
    # print percent done
    print(i / len(short_description))
    # limit to 512 tokens
    encoded_input = tokenizer(short_description[i:i + batch_size], return_tensors='pt', padding=True, truncation=True,
                              max_length=512).to(device)
    # limit to 512 tokens
    output = model(**encoded_input)
    output = output[1].cpu().detach().numpy()
    outputs.append(output)
    if output.shape not in shapes:
        shapes.append(output.shape)

print(shapes)

# recombine the outputs
outputs = np.concatenate(outputs, axis=0)

# save the outputs
with open('data/processed/short_description_bert_outputs_cleaned.npy', 'wb') as f:
    np.save(f, outputs)
