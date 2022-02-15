from keras import Model
from keras.layers import Input, LSTM, Conv1D, Conv2D, Reshape, Dense, BatchNormalization, Dropout, concatenate

def build_model(window_size, depth):

    input_layer1 = Input(shape=(window_size, depth * 4, 1))
    # input_layer2 = Input(shape=(window_size, n_features))

    x = Conv2D(16, kernel_size=(1, 2), strides=(1, 2), activation='relu')(input_layer1)
    x = Conv2D(16, kernel_size=(4, 1), activation='relu', padding='same')(x)
    x = Conv2D(16, kernel_size=(4, 1), activation='relu', padding='same')(x)

    x = Conv2D(16, kernel_size=(1, 2), strides=(1, 2), activation='relu')(x)
    x = Conv2D(16, kernel_size=(4, 1), activation='relu', padding='same')(x)
    x = Conv2D(16, kernel_size=(4, 1), activation='relu', padding='same')(x)

    x = Conv2D(16, kernel_size=(1, depth), activation='relu')(x)
    x = Conv2D(16, kernel_size=(4, 1), activation='relu', padding='same')(x)
    x = Conv2D(16, kernel_size=(4, 1), activation='relu', padding='same')(x)

    x = Reshape((window_size, 16))(x)

    # x = concatenate([x, input_layer2])

    lstm_layer = LSTM(64)(x)
    bn_layer = BatchNormalization()(lstm_layer)
    dropout_layer = Dropout(0.8)(bn_layer)
    output_layer = Dense(3, activation='softmax')(dropout_layer)

    # model = Model([input_layer1, input_layer2], output_layer)
    model = Model(input_layer1, output_layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model