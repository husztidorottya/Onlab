from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
import argparse
import numpy as np


def main():
    letters = dict()
    x_train = [] 
    Y_train = []

    parser = argparse.ArgumentParser()
    parser.add_argument('category_one_inputname')
    parser.add_argument('category_two_inputname')
    args = parser.parse_args()

    with open(args.category_one_inputname) as file:
        for line in file:
            coded_word = []
            for character in line:
                index = letters.setdefault(character,len(letters)+1)
                coded_word.append(index)
            Y_train.append(0)
            x_train.append(coded_word)

    #leghosszabb szo megtalalasa    
    max_length = max(x_train,key=len)

    j = 0
    
    #referenciahossznak megfelelo padding kepzese
    for word in x_train:
        if len(word)!=max_length:
            for i in range(0,len(max_length)-len(word)):
                x_train[j].append(0)
        j += 1        
    
    X_train = np.array(x_train)

    #max ennyifele erteket vehet fel
    number = len(letters)+1

    model = Sequential()
    model.add(Embedding(number, 10))
    model.add(LSTM(output_dim = 10, activation = 'sigmoid', inner_activation = 'hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
    earlyStopping = EarlyStopping(monitor='val_loss',patience=0,verbose=0,mode='auto')
    model.fit(X_train, Y_train, batch_size = 16, nb_epoch = 10, callbacks=[earlyStopping],validation_split=0.1)

if __name__ == '__main__':
    main()
