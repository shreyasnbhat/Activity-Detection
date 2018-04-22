from keras import backend as K
from keras.layers import Conv2D, Dropout, LSTM, BatchNormalization, Input,Activation, MaxPool2D, Flatten, Dense, TimeDistributed, Conv1D, Lambda 
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers.convolutional import ZeroPadding2D

def end_to_end(input_shape):
    X_input = Input(input_shape)
    X = TimeDistributed(BatchNormalization(), name = 'BatchNorm_1')(X_input)
    X = TimeDistributed(Conv2D(32, (7, 7), strides = (4, 4), activation='relu', padding="same"), name="Conv_1a")(X)
    X = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding="same") , name="Conv_1b")(X)
    X = TimeDistributed(MaxPool2D((2, 2)),  name = "Pool_1")(X)
    X = TimeDistributed(Dropout(0.2), name = "Dropout_1")(X)
    
    X = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding = "same"), name ="Conv_2a")(X)
    X = TimeDistributed(MaxPool2D((2, 2)), name = "Pool_2")(X)
    X = TimeDistributed(Dropout(0.2), name= "Dropout_2")(X)
    
    X = TimeDistributed(Conv2D(32,(3,3), name='Conv_3a', activation='relu'))(X)
    X = TimeDistributed(MaxPool2D((2, 2), name = "Pool_3"))(X)
    X = TimeDistributed(Dropout(0.2), name = "Dropout_3")(X)

    X = TimeDistributed(Conv2D(8,(1,1), activation='relu'), name='Conv_1x1')(X)
    X = TimeDistributed(Flatten())(X)
    X = TimeDistributed(Dropout(0.3))(X)
    Y = TimeDistributed(Dense(9,activation='softmax',name='final'),name='Secondary_Output')(X)
    
    X = LSTM(48, return_sequences=False,dropout=0.3, name = "LSTM1")(X)
    #X = LSTM(32, return_sequences=False)(X)
    X = Dense(9, activation='softmax',name='Primary_Output')(X)

    return Model(X_input, outputs=[X, Y])

def dilated_conv(input_shape):
    X_input = Input(input_shape)
    X = TimeDistributed(BatchNormalization(), name = 'BatchNorm_1')(X_input)
    #X = TimeDistributed(ZeroPadding2D((3, 3)))(X)
    X = TimeDistributed(Conv2D(32, (7, 7), strides = (4, 4), activation='relu', padding="same"), name="Conv_1a")(X)
    X = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding="same"), name="Conv_1b")(X)
    X = TimeDistributed(MaxPool2D((2, 2)), name = "Pool_1")(X)
    X = TimeDistributed(Dropout(0.2), name='Dropout_a')(X)
    
    X = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding = "same"), name ="Conv_2a")(X)
    #X = TimeDistributed(Conv2D(32, (3, 3), name ="Conv_2b", activation='relu', padding = "same"))(X)
    X = TimeDistributed(MaxPool2D((2, 2)), name = "Pool_2")(X)
    X = TimeDistributed(Dropout(0.2), name='Dropout_b')(X)
    X = TimeDistributed(Conv2D(32,(3,3),activation='relu'), name='Conv_3a')(X)
    X = TimeDistributed(MaxPool2D((2, 2)), name = "Pool_3")(X)
    
    X = TimeDistributed(Conv2D(8,(1,1),activation='relu'), name='Conv_1x1')(X)
    X = TimeDistributed(Flatten(), name='Flatten')(X)
    X = TimeDistributed(Dropout(0.3), name='Dropout_c')(X)
    Y = TimeDistributed(Dense(9,activation='softmax',name='final'))(X)

    X = Conv1D(64, 4, dilation_rate=2, name = 'Conv1Da', activation='relu')(X)
    X = Conv1D(48, 3, dilation_rate=4, name = 'Conv1Db', activation='relu')(X)
    X = Conv1D(32, 3, dilation_rate=4, name = 'Conv1Dc', activation='relu')(X)
    X = Lambda(lambda x : x[:, -1, :], name = "Extractoutput")(X)
    X = Dense(9, activation='softmax', name = 'Output')(X)
    return Model(X_input, [X,Y])
