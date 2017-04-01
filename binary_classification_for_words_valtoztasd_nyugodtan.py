from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import re
import sys

"""
Code words by letters according to one-hot-encoding

Variables:
   - coded_word: contains a word encoded by one-hot-encoding
   - letters: dictionary, where key is the letter and value is one-hot-encoded value of the letter
   - index: value of the letter
   - Y_set: all labels
   - x_set: all data set
   - category: word’s label
"""
def code_word(category, data, x_set, Y_set, letters):
    for line in data:
            coded_word = []
            for character in line:
                index = letters.setdefault(character, len(letters)+1)
                coded_word.append(index)
            Y_set.append(category)
            x_set.append(coded_word)



"""
Create padding according to the maximum length in the x_set

Variables:
   - max_length: maximum length of words in all data set
"""
def create_padding(x_set):
    # find maximum length    
    max_length = max(x_set, key=len)  

    j = 0
    
    # create padding according to max_length
    for word in x_set:
        if len(word) != max_length:
            for i in range(0, len(max_length) - len(word)):
                x_set[j].append(0)
        j += 1  



# Main function
def main():
    letters = dict()
    x_train = [] 
    x_set = []
    Y_set = []
    x_test = []
    Y_train = []
    Y_test = []
    """
    # read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('category_one_inputname')
    parser.add_argument('category_two_inputname')
    args = parser.parse_args()
    """
    derivations = input('Derivaciok:')
    derivations = derivations.replace('\t','[^/]*').replace(';','[^a-zA-Z]')
    rules = '^' + derivations + '[^/]*$'
    patt = re.compile(rules)

    filename = 'webcorp.emmorph.nondisambig.1Mtokens'
  
    previous_empty_line = True
    data = []

    with open(filename) as input_file:
        for line in input_file:
                if line!='\n' and previous_empty_line:
                        if line.find('/')!=-1:
                                m = patt.match(line)
                                if m:
                                        splited_line = line.replace('[',';;;').replace(']',';;;').replace('\t',';;;').split(';;;')

                                        if splited_line[0] not in data:
                                                data.append(splited_line[0])


                                previous_empty_line = False

                if line=='\n':
                        previous_empty_line = True
  
    # encoding words and append to dataset and labels
    code_word(0, data, x_set, Y_set, letters)

    derivations = input('Derivaciok:')
    derivations = derivations.replace('\t','[^/]*').replace(';','[^a-zA-Z]')
    rules = '^' + derivations + '[^/]*$'
    patt = re.compile(rules)

    previous_empty_line = True
    data2 = []

    with open(filename) as input_file:
        for line in input_file:
                if line!='\n' and previous_empty_line:
                        if line.find('/')!=-1:
                                m = patt.match(line)
                                if m:
                                        splited_line = line.replace('[',';;;').replace(']',';;;').replace('\t',';;;').split(';;;')

                                        if splited_line[0] not in data2:
                                                data2.append(splited_line[0])


                                previous_empty_line = False

                if line=='\n':
                        previous_empty_line = True

    # encoding words and append to dataset and labels
    code_word(1, data2, x_set, Y_set, letters)

    create_padding(x_set)

    # split randomly x_set to train and test dataset (90% train + 10% test)
    x_train, x_test, Y_train, Y_test = train_test_split(x_set, Y_set, test_size = 0.1)

    X_train = np.array(x_train)
    X_test = np.array(x_test)
    
    # max number of values
    number = len(letters)+1

    # defining model
    model = Sequential()
    model.add(Embedding(number, 100))
    model.add(LSTM(output_dim = 100, activation = 'sigmoid', inner_activation = 'hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # train and test, using earlyStopping to stop after 1 rounds if not improving
    model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
    earlyStopping = EarlyStopping(monitor = 'val_loss', patience = 1, verbose = 0, mode = 'auto')
    model.fit(X_train, Y_train, batch_size = 16, nb_epoch = 10, callbacks = [earlyStopping], validation_split = 0.1)
    score = model.evaluate(X_test, Y_test, batch_size = 16)
    print(score)

if __name__ == '__main__':
    main()
