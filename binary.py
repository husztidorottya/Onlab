from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
import argparse
import numpy as np

def main():
    letters = dict()
    X_train = np.array([])
    Y_train = []

    parser = argparse.ArgumentParser()
    parser.add_argument('category_one_inputname')
    parser.add_argument('category_two_inputname')
    args = parser.parse_args()

    with open(args.category_one_inputname) as category_one_input:
        for line in category_one_input:
            coded_word = []
            for character in line:
                index = letters.setdefault(character,len(letters))
                character_index = []
                character_index.append(index)
                coded_word.append(character_index)
            X_train = np.append(X_train,coded_word)
            Y_train.append(0)

    number = len(Y_train)

    model = Sequential()
    model.add(Embedding(number, 10, input_length =1))
    model.add(LSTM(output_dim = 10, activation = 'sigmoid', inner_activation = 'hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
    model.fit(X_train, Y_train, batch_size = 16, nb_epoch = 10)

if __name__ == '__main__':
    main()
