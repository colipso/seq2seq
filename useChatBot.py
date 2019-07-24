from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.layers import Input, LSTM, Dense
import numpy as np

latent_dim = 256

def loadData(resultFile = './model_chatBot/data.txt'):
    with open(resultFile,'rb') as f:
        result = {}  
        result = pickle.load(f)
    return result

result = loadData()

num_encoder_tokens = result['num_encoder_tokens']
num_decoder_tokens = result['num_decoder_tokens']
#encoder_input_data = result['encoder_input_data']
#decoder_input_data = result['decoder_input_data']
#decoder_target_data = result['decoder_target_data']
max_decoder_seq_length = result['max_decoder_seq_length']
input_token_index = result['input_token_index']
target_token_index = result['target_token_index']
#input_texts = result['input_texts']
max_encoder_seq_length = result['max_encoder_seq_length']


model = load_model('./model_chatBot/s2s_chatBot.h5')
encoder_inputs = model.input[0]
encoder_outputs, state_h, state_c = model.layers[2].output
encoder_states = [state_h, state_c]
# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,),name='input_3')
decoder_state_input_c = Input(shape=(latent_dim,),name='input_4')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_lstm = model.layers[3]
decoder_inputs = model.input[1]
decoder_dense = model.layers[4]

decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
                len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

def encode_input_data(sentence='你是谁'):
    encoder_input_data = np.zeros(
        (1, max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')
    for t, char in enumerate(sentence):
        encoder_input_data[0, t, input_token_index[char]] = 1.
    return encoder_input_data

testList = ['我该怎么办','我因为工作而烦恼','办理不完怎么办','你有没有朋友','是谁呀','是和你聊天的人吗','今晚吃什么']
for seq_index in range(len(testList)):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encode_input_data(testList[seq_index])
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', testList[seq_index])
    print('Decoded sentence:', decoded_sentence)
