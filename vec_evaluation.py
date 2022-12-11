# we want to see if the custom vec can be predicted by the bert vecs, so we will train simple linear models and
# compare the MSE of the predictions

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import systems
from data_to_custom_vec import make_vecs


def train_and_evaluate(vecs, labels):
    """Train a linear model on the vecs and evaluate the MSE on the labels."""
    model = LinearRegression()
    model.fit(vecs, labels)
    predictions = model.predict(vecs)
    return mean_squared_error(labels, predictions)

def main():
    bert_vecs = [systems.load_bert_vec(filename) for filename in systems.bert_filenames]

    # create the custom vecs
    custom_vecs, h = make_vecs()

    # train and evaluate the models
    for filename, bert_vec in zip(systems.bert_filenames, bert_vecs):
        print(f'filename: {filename}')
        print(f'mse: {train_and_evaluate(bert_vec, custom_vecs)}')
        print()

    # create a final model by combining all the bert vecs
    bert_vecs = np.concatenate(bert_vecs, axis=1)
    print('all bert vecs combined')
    print(f'mse: {train_and_evaluate(bert_vecs, custom_vecs)}')



if __name__ == '__main__':
    main()
