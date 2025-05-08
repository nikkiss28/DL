from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import SimpleRNN,Dense,Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
max_words=10000
max_len=100
(X_train,y_train),(X_test,y_test)=imdb.load_data(num_words=max_words)
X_train=pad_sequences(X_train,maxlen=max_len)
X_test=pad_sequences(X_test,maxlen=max_len)
model=Sequential()
model.add(Embedding(input_dim=max_words,output_dim=64,input_length=max_len))
model.add(SimpleRNN(32))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=5,batch_size=64,validation_data=[X_test,y_test])
loss,acc=model.evaluate(X_test,y_test)
print(acc)
