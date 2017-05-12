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
Shifting along the word with ngrams n

Variables:
    - ngrams_n: how many characters are coded every time
    - ngram: n characters together (window which is shifted)
"""
def code_ngrams(begin, end, ngrams_n, line, letters, coded_word):
    for char in range(begin, end):
        ngram = ''
        # get ngram
        for i in range(0, ngrams_n):
            if ((char + i) < len(line)) & ((char + i) < end):
                ngram += line[char + i]
            else:
                break
        index = letters.setdefault(ngram, len(letters) + 1)
        coded_word.append(index)


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
def code_word(data, x_set, Y_set, letters, char_num, char_dir, ngrams_enabled, ngrams_n, test_cat):
    line_num = 0
    for line in data:
            coded_word = []
            
            # word length equals or less than taken characters
            if len(line) <= char_num:
                # ngrams disabled
                if ngrams_enabled == 0:
                    for character in line:
                        index = letters.setdefault(character, len(letters) + 1)
                        coded_word.append(index)
                
                # ngrams enabled
                else:
                    # shifting a window along the taken number of characters
                    code_ngrams(0, len(line) - ngrams_n, ngrams_n, line, letters, coded_word)
                
                    
            # word is longer than taken characters
            else:
                
                # take characters from the beginning of the word
                if char_dir == 0:
                    
                    # ngrams disabled
                    if ngrams_enabled == 0:
                        # take char_num characters from the beginning
                        for char in range(0, char_num):
                            index = letters.setdefault(line[char], len(letters) + 1)
                            coded_word.append(index)
                    
                    # ngrams enabled
                    else:
                        # shifting a window along the taken number of characters
                        code_ngrams(0, char_num, ngrams_n, line, letters, coded_word)
                    
                # take characters from the end of the word
                if char_dir == 1:
                    
                    # ngrams disabled
                    if ngrams_enabled == 0:
                        # take char_num characters from the end
                        for char in range((len(line) - char_num), len(line)):
                            index = letters.setdefault(line[char], len(letters) + 1)
                            coded_word.append(index)
                    
                    # ngrams enabled
                    else:
                        # shifting a window along the taken number of characters
                        code_ngrams((len(line) - char_num), len(line), ngrams_n, line, letters, coded_word)
            Y_set.append(test_cat[line_num])
            x_set.append(coded_word)
            line_num += 1



"""
Create padding according to the maximum length in the x_set

Variables:
   - max_length: maximum length of words in all data set
"""
def create_padding(x_set, char_dir):
    # find maximum length    
    max_length = max(x_set, key=len)  

    j = 0
   
    # create padding according to max_length
    for word in x_set:
        padding_word = []
        if len(word) != max_length:
            for i in range(0, len(max_length) - len(word)):
                # create padding taking characters from the beginning
                if char_dir == 0:
                    x_set[j].append(0)
                # create padding taking characters from the back
                else:
                    padding_word.append(0)
            if char_dir == 1:
                word = x_set[j]
                x_set[j] = padding_word + word
                
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
def experiment(number, X_train, Y_train, X_test, Y_test, embedding_dimension, LSTM_cellnum, batch_size_num, nb_epoch_num, char_num, char_dir, ngrams_enabled, ngrams_n, test_words, test_cat):
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
    log_experiment(embedding_dimension, LSTM_cellnum, batch_size_num, nb_epoch_num, score, char_num, char_dir, ngrams_enabled, ngrams_n)

    # show bad predictions
    show_mispredictions(test_words, test_cat, X_test, Y_test, model)
   
"""
Choosing parameters randomly
"""
def choose_parameters_randomly(embedding_dimension_paramvalues, LSTM_cellnum_paramvalues, batch_size_num_paramvalues, char_num_paramvalues, char_dir_paramvalues, ngrams_enabled_paramvalues, ngrams_n_paramvalues):
    #embedding_dimension = random.choice(embedding_dimension_paramvalues)
    #LSTM_cellnum = random.choice(LSTM_cellnum_paramvalues)
    #batch_size_num = random.choice(batch_size_num_paramvalues)
    #char_num = random.choice(char_num_paramvalues)
    #char_dir = random.choice(char_dir_paramvalues)
    #ngrams_enabled = random.choice(ngrams_enabled_paramvalues)
    #ngrams_n = random.choice(ngrams_n_paramvalues)
   
    embedding_dimension = 50
    LSTM_cellnum = 16
    batch_size_num = 4
    char_num = 10
    char_dir = 1
    ngrams_enabled = 0
    ngrams_n = 3

    return (embedding_dimension, LSTM_cellnum, batch_size_num, char_num, char_dir, ngrams_enabled, ngrams_n)


"""
Logging the experiment parameters and values to a file
"""
def log_experiment(embedding_dimension, LSTM_cellnum, batch_size_num, nb_epoch_num, score, char_num, char_dir, ngrams_enabled, ngrams_n):
    with open('log_experiment.tsv','a') as logfile:
        logfile.write('Embedding_dimension\t{0}\n\n'.format(embedding_dimension))
        logfile.write('LSTM_cellnum\t{0}\n'.format(LSTM_cellnum))
        logfile.write('Nb_epoch\t{0}\n'.format(nb_epoch_num))
        logfile.write('Batch_size\t{0}\n'.format(batch_size_num))
        logfile.write('Score\t{0}\n'.format(score))
        logfile.write('Taken_characters\t{0}\n'.format(char_num))
        logfile.write('Taken_characters_dir\t{0}\n'.format(char_dir))
        logfile.write('Ngrams_enabled\t{0}\n'.format(ngrams_enabled))
        logfile.write('Ngrams_n\t{0}\n'.format(ngrams_n))

        
"""
Log mispredications which are different from expected
"""
def show_mispredictions(test_words, test_cat, X_test, Y_test, model):
    predictions = model.predict_classes(X_test, batch_size = 32, verbose = 0)
    with open('mispredictions.tsv', 'w') as file:
        for i in range(0,len(predictions)):
            if predictions[i] != Y_test[i]:
                # word - expected - prediction
                file.write('{0}\t{1}\t{2}\n'.format(test_words[i],Y_test[i],predictions[i]))

# Main function
def main():
    # define parameter values
    embedding_dimension_paramvalues = [1,10,20,30,40,50]
    LSTM_cellnum_paramvalues = [2,4,8,16,32,64,128,256,512]
    batch_size_num_paramvalues = [2,4,8,16,32,64,128,256,512]
    nb_epoch_num = 500
    # take x characters from the end of the word
    char_num_paramvalues = [1,2,3,4,5,6,7,8,9,10]
    # take characters from the beginning or the end of the word (0 - beginning, 1 - end)
    char_dir_paramvalues = [0,1]
    # use other ngrams or not (0 - disabled, 1 - enabled)
    ngrams_enabled_paramvalues = [0,1]
    # N value of Ngrams (how many characters are coded together)
    ngrams_n_paramvalues = [1,2,3]
    
    # read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_number')
    args = parser.parse_args()
    
    # input filename
    filename = 'webcorp.emmorph.nondisambig.1Mtokens'
    
    # derivations format eg.:	tab/N	tab/N
    derivations_format1 = input('Derivaciok:')
    derivations_format2 = input('Derivaciok:') 
    
    # write the number of experiments and derivations to the log file
    with open('log_experiment.tsv','w') as logfile:
         logfile.write('Experiment_num\t{0}\n'.format(args.experiment_number))
         logfile.write('Derivations1{0}\n'.format(derivations_format1))
         logfile.write('Derivations2{0}\n'.format(derivations_format2))


    # doing given number of experiments
    for round in range(0,int(args.experiment_number)):
        letters = dict()
        x_train = [] 
        x_set = []
        Y_set = []
        x_test = []
        Y_train = []
        Y_test = []
        categories = []
        words = []
    
        # choosing parameters randomly
        embedding_dimension, LSTM_cellnum, batch_size_num, char_num, char_dir, ngrams_enabled, ngrams_n = choose_parameters_randomly(embedding_dimension_paramvalues, LSTM_cellnum_paramvalues, batch_size_num_paramvalues, char_num_paramvalues, char_dir_paramvalues, ngrams_enabled_paramvalues, ngrams_n_paramvalues)

        # handle the two derivations
        for i in range(0,2):
            if i == 0:
                derivations = derivations_format1.replace('\t','[^/]*').replace(';','[^a-zA-Z]')
            else:
                derivations = derivations_format2.replace('\t','[^/]*').replace(';','[^a-zA-Z]')
            rules = '^' + derivations + '[^/]*$'
            patt = re.compile(rules)
      
            # words
            data = []

            #filter the derivation type of words
            filter_derivations(filename,data,patt)
  
            words = words + data
            for j in range(0,len(data)):
                if i == 0:
                    categories.append(0)
                else:
                    categories.append(1)

        train_words, test_words, train_cat, test_cat = train_test_split(words, categories, test_size = 0.1)

        code_word(train_words, x_train, Y_train, letters, char_num, char_dir, ngrams_enabled, ngrams_n, train_cat)
        code_word(test_words, x_test, Y_test, letters, char_num, char_dir, ngrams_enabled, ngrams_n, test_cat)
  
        create_padding(x_train, char_dir)
        create_padding(x_test, char_dir)

        X_train = np.array(x_train)
        X_test = np.array(x_test)
    
        # max number of values
        number = len(letters)+1
        
        # start learning
        experiment(number,X_train,Y_train,X_test,Y_test,embedding_dimension,LSTM_cellnum,batch_size_num,nb_epoch_num, char_num, char_dir, ngrams_enabled, ngrams_n, test_words, test_cat)    
        
        

if __name__ == '__main__':
    main()

