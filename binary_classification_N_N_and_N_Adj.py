#
# Bináris osztályozó neurális háló, amely eldönti a szóalak alapján, hogy N-N vagy N-Adj deriváció van benne
#

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from collections import defaultdict

N = 0
M = 0

words = defaultdict(int)
Y_train = []
X_train = []

category_one_inputname = input('Elso kategoria forrasfajlja:')
category_two_inputname = input('Masik kategoria forrasfajlja:')

i = 0

with open(category_one_inputname) as category_one_input:
	for line in category_one_input:
		N += 1
		if words[line]==0:
			words[line] = i
			X_train.append(i)
			Y_train.append(0)
			i += 1

with open(category_two_inputname) as category_two_input:
	for line in category_two_input:
		M += 1
		if words[line]==0:
			words[line] = i
			X_train.append(i)
			Y_train.append(1)
			i += 1
number = len(Y_train)


model = Sequential()
model.add(Embedding(number,3, input_length=1))
model.add(LSTM(output_dim=3,activation='sigmoid',inner_activation='hard_sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

model.fit(X_train,Y_train, batch_size=16, nb_epoch=10)

