from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
import numpy as np
import time
import pprint
import pickle
from tensorflow.keras.callbacks import TensorBoard
import random

batch_size = 64  # Batch size for training.
epochs = 50  # Number of epochs to train for.
steps_per_epoch = 1000
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 770000  # Number of samples to train on.
# Path to the data txt file on disk.
def loadData(inputData = './data/train/out.txt' , targetData = './data/train/in.txt' , resultFile = './model/data.txt'):
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()
    with open(inputData,'r',encoding='utf-8') as f:
        num_samples = len(f.readlines())
    with open(inputData,'r',encoding='utf-8') as f:
        lines = f.read().split('\n')
    for line in lines[:num_samples]:
        input_text = line.replace(' ','')
        input_texts.append(input_text)
        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)

    with open(targetData,'r',encoding='utf-8') as f:
        lines = f.read().split('\n')
    for line in lines[:num_samples]:
        line = '\t' +' '+ line +' '+'\n'
        target_text = line.replace(' ','')
        target_texts.append(target_text)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)

    input_characters = sorted(list(input_characters | target_characters))
    target_characters = input_characters
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    print('Number of samples:', len(input_texts))
    print('Number of unique input tokens:', num_encoder_tokens)
    print('Number of unique output tokens:', num_decoder_tokens)
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)

    input_token_index = dict(
        [(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict(
        [(char, i) for i, char in enumerate(target_characters)])
    
    result = {'num_encoder_tokens':num_encoder_tokens,
            'num_decoder_tokens':num_decoder_tokens,
            #'encoder_input_data':encoder_input_data,
            #'decoder_input_data':decoder_input_data,
            #'decoder_target_data':decoder_target_data,
            'input_token_index':input_token_index,
            'target_token_index':target_token_index,
            'max_decoder_seq_length':max_decoder_seq_length,
            'max_encoder_seq_length':max_encoder_seq_length,
            'input_texts':input_texts,
            'target_texts':target_texts,
            'num_samples':num_samples
            }
    savedResult = {'num_encoder_tokens':num_encoder_tokens,
            'num_decoder_tokens':num_decoder_tokens,
            #'encoder_input_data':encoder_input_data,
            #'decoder_input_data':decoder_input_data,
            #'decoder_target_data':decoder_target_data,
            'input_token_index':input_token_index,
            'target_token_index':target_token_index,
            'max_decoder_seq_length':max_decoder_seq_length,
            'max_encoder_seq_length':max_encoder_seq_length,
            #'input_texts':input_texts
            }
    with open(resultFile,'wb') as f:
        pickle.dump(savedResult, f)  #read d = {}  d = pickle.load(f)
    
    return result

#result = loadData()
def data_generator(data, forValidation=False): #data=result
    while True:
        max_encoder_seq_length = data['max_encoder_seq_length']
        num_encoder_tokens = data['num_encoder_tokens']
        max_decoder_seq_length = data['max_decoder_seq_length']
        num_decoder_tokens = data['num_decoder_tokens']
        input_token_index = data['input_token_index']
        target_token_index = data['target_token_index']
        input_texts = data['input_texts']
        target_texts = data['target_texts']

        '''
        if forValidation:
            j = random.randint(700000,770000)
        else:
            j = random.randint(1,700000)
        '''
        if forValidation:
            j = random.randint( int(data['num_samples']*0.9) ,int(data['num_samples']) )
        else:
            j = random.randint( 1 , int(data['num_samples']*0.9) )
        encoder_input_data = np.zeros(
            (batch_size, max_encoder_seq_length, num_encoder_tokens),
            dtype='float32')
        decoder_input_data = np.zeros(
            (batch_size, max_decoder_seq_length, num_decoder_tokens),
            dtype='float32')
        decoder_target_data = np.zeros(
            (batch_size, max_decoder_seq_length, num_decoder_tokens),
            dtype='float32')
        
        #one_hot code
        for i, (input_text, target_text) in enumerate(zip(input_texts[j:j+batch_size], target_texts[j:j+batch_size])):
            for t, char in enumerate(input_text):
                #print(i)
                encoder_input_data[i, t, input_token_index[char]] = 1.
            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_input_data[i, t, target_token_index[char]] = 1.
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[i, t - 1, target_token_index[char]] = 1.
        #print("===J: " + str(j))
        #print("===Input: " + input_texts[j])
        #print("===Target: " + target_texts[j])
        yield [encoder_input_data, decoder_input_data], decoder_target_data

tbCallBack = TensorBoard(
    log_dir='./logs',
    histogram_freq=1,
    batch_size=32,
    write_graph=True,
    write_grads=True,
    write_images=True
)

def trainSeq2SeqModel(inputData = './data/train/out.txt' , 
                    targetData = './data/train/in.txt' , 
                    resultFile = './model/data.txt', 
                    saveFile = './model/s2s.h5',
                    steps_per_epoch = steps_per_epoch,
                    validation_steps = 10,
                    epochs=epochs
                    ):
    result = loadData(inputData = inputData , targetData = targetData , resultFile = resultFile)
    num_encoder_tokens = result['num_encoder_tokens']
    num_decoder_tokens = result['num_decoder_tokens']
    #encoder_input_data = result['encoder_input_data']
    #decoder_input_data = result['decoder_input_data']
    #decoder_target_data = result['decoder_target_data']
    #max_decoder_seq_length = result['max_decoder_seq_length']
    #input_token_index = result['input_token_index']
    #target_token_index = result['target_token_index']
    #input_texts = result['input_texts']
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                        initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Run training
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit_generator(
            generator=data_generator(data = result),
            validation_data=data_generator(data = result,forValidation=True),
            steps_per_epoch=steps_per_epoch,
            validation_steps = 10,
            #batch_size=batch_size,
            epochs=epochs,
            #validation_split=0.2,
            callbacks=[tbCallBack])
    # Save model
    model.save(saveFile)

if __name__ == "__main__":
    trainSeq2SeqModel()

