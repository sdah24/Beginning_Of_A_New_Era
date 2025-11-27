import numpy as np
import tensorflow as tf
from pygments.lexer import words
from sympy import sequence
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentence = ["I love cat","I love flower","you love Cat!","Do you think my cat is beautiful?"]

#initializing Tokenizer
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")

#tokenizing each word of 'sentence' list, using oov for unknown word in for future use cases.....see test_seq for example
tokenizer.fit_on_texts(sentence)

#indexing each word
widx =  tokenizer.word_index

#making sequence of tokens(numbers) to help the machine to understand better
sequences = tokenizer.texts_to_sequences(sentence)

#using padding for fitting all the sequence to match the sequence with highest length
padded = pad_sequences(sequences,padding="post")

print(widx)
print(sequences)
print(padded)

#testing sentence sequence without tokenizing them first
test_data =  ['i really love eating',"i also love sleeping"]

test_seq =  tokenizer.texts_to_sequences(test_data)

print(test_seq)