from simpleSeq2Seq import *
batch_size = 64  # Batch size for training.
epochs = 150  # Number of epochs to train for.
steps_per_epoch = 1000
latent_dim = 256  # Latent dimensionality of the encoding space.
trainSeq2SeqModel(
    inputData='./chatBotData/question.txt',
    targetData='./chatBotData/answer.txt',
    resultFile='./model_chatBot/data.txt',
    saveFile='./model_chatBot/s2s_chatBot.h5',
    steps_per_epoch = steps_per_epoch,
    validation_steps = 10,
    epochs=epochs
)
