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
import random

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



"""
Filter specific derivations from file

Variables:
    - previous_empty_line: if it is a block beginning
    - filename: input filename
    - line: one line of the input file
    - patt: regular expression
    - splited_line: contains splitted line, 0. element is the word
    - data: contains specific type of derivations’ words
"""
def filter_derivations(filename, data, patt):
    # if it is a block beginning
    previous_empty_line = True

    # read the input file
    with open(filename) as input_file:
        for line in input_file:
                # if not empty line and block beginning 
                if line != '\n' and previous_empty_line:
                        # if there’s derivation in the line
                        if line.find('/') != -1:
                                m = patt.match(line)
                                # matching regular expression
                                if m:
                                        splited_line = line.replace('[',';;;').replace(']',';;;').replace('\t',';;;').split(';;;')

                                        # if the word is not already in the dataset
                                        if splited_line[0] not in data:
                                                data.append(splited_line[0])

                                # finish this block, we need only one element of a block
                                previous_empty_line = False

                # if empty line
                if line == '\n':
                        previous_empty_line = True



"""
Doing the experiment

"""
def experiment(number, X_train, Y_train, X_test, Y_test, embedding_dimension_paramvalues, LSTM_cellnum_paramvalues, batch_size_num_paramvalues, nb_epoch_num_paramvalues):
    # choosing parameters randomly
    embedding_dimension, LSTM_cellnum, batch_size_num, nb_epoch_num = choose_parameters_randomly(embedding_dimension_paramvalues, LSTM_cellnum_paramvalues, batch_size_num_paramvalues, nb_epoch_num_paramvalues)

    # defining model
    model = Sequential()
    model.add(Embedding(number, embedding_dimension))
    model.add(LSTM(output_dim = LSTM_cellnum, activation = 'sigmoid', inner_activation = 'hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # train and test, using earlyStopping to stop after 1 rounds if not improving
    model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
    earlyStopping = EarlyStopping(monitor = 'val_loss', patience = 1, verbose = 0, mode = 'auto')
    model.fit(X_train, Y_train, batch_size = batch_size_num, nb_epoch = nb_epoch_num, callbacks = [earlyStopping], validation_split = 0.1)
    score = model.evaluate(X_test, Y_test, batch_size = batch_size_num)
    print(score)

    # log the parameters and result of the experiment
    log_experiment(embedding_dimension, LSTM_cellnum, batch_size_num, nb_epoch_num, score)


"""
Choosing parameters randomly
"""
def choose_parameters_randomly(embedding_dimension_paramvalues, LSTM_cellnum_paramvalues, batch_size_num_paramvalues, nb_epoch_num_paramvalues):
    embedding_dimension = random.choice(embedding_dimension_paramvalues)
    LSTM_cellnum = random.choice(LSTM_cellnum_paramvalues)
    batch_size_num = random.choice(batch_size_num_paramvalues)
    nb_epoch_num = random.choice(nb_epoch_num_paramvalues)

    return (embedding_dimension, LSTM_cellnum, batch_size_num, nb_epoch_num)


"""
Logging the experiment parameters and values to a file
"""
def log_experiment(embedding_dimension, LSTM_cellnum, batch_size_num, nb_epoch_num, score):
    with open('log_experiment.tsv','a') as logfile:
        logfile.write('Embedding_dimension\t{0}\n'.format(embedding_dimension))
        logfile.write('LSTM_cellnum\t{0}\n'.format(LSTM_cellnum))
        logfile.write('Nb_epoch\t{0}\n'.format(nb_epoch_num))
        logfile.write('Batch_size\t{0}\n'.format(batch_size_num))
        logfile.write('Score\t{0}\n\n'.format(score))
       


# Main function
def main():
    letters = dict()
    x_train = [] 
    x_set = []
    Y_set = []
    x_test = []
    Y_train = []
    Y_test = []
    
    # define parameter values
    embedding_dimension_paramvalues = []
    embedding_dimension_paramvalues.append(1)
    # embedding dimension parameter values generated (1,10,20,30,...,100)
    for i in range(1,11):
        embedding_dimension_paramvalues.append(i*10)

    LSTM_cellnum_paramvalues = [2,4,8,16,32,64,128,256,512]
    batch_size_num_paramvalues = [2,4,8,16,32,64,128,256,512]
    nb_epoch_num_paramvalues = []
    # nb_epoch parameter values generated (10,20,30,...,100)
    for i in range(1,11):
        nb_epoch_num_paramvalues.append(i*10)

    # read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_number')
    args = parser.parse_args()    

    # input filename
    filename = 'webcorp.emmorph.nondisambig.1Mtokens'

    # handle the two derivations
    for i in range(0,2):
        # derivations format eg.:	tab/N	tab/N
        derivations_format = input('Derivaciok:')
        derivations = derivations_format.replace('\t','[^/]*').replace(';','[^a-zA-Z]')
        rules = '^' + derivations + '[^/]*$'
        patt = re.compile(rules)
      
        # words
        data = []

        #filter the derivation type of words
        filter_derivations(filename,data,patt)
  
        if i == 0:
            # encoding words and append to dataset and labels
            code_word(0, data, x_set, Y_set, letters)
            derivations_format1 = derivations_format
        else:
            # encoding words and append to dataset and labels
            code_word(1, data, x_set, Y_set, letters)
            derivations_format2 = derivations_format

    create_padding(x_set)

    # split randomly x_set to train and test dataset (90% train + 10% test)
    x_train, x_test, Y_train, Y_test = train_test_split(x_set, Y_set, test_size = 0.1)

    X_train = np.array(x_train)
    X_test = np.array(x_test)
    
    # max number of values
    number = len(letters)+1

    # write the number of experiments and derivations to the log file
    with open('log_experiment.tsv','w') as logfile:
        logfile.write('Experiment_num\t{0}\n'.format(args.experiment_number))
        logfile.write('Derivations1{0}\n'.format(derivations_format1))
        logfile.write('Derivations2{0}\n'.format(derivations_format2))

    # doing given number of experiments
    for round in range(0,int(args.experiment_number)):
       experiment(number,X_train,Y_train,X_test,Y_test,embedding_dimension_paramvalues,LSTM_cellnum_paramvalues,batch_size_num_paramvalues,nb_epoch_num_paramvalues)    
    

if __name__ == '__main__':
    main()
