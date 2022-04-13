import pandas as pd
import os
import joblib
import yaml
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import re
import glob

from makers.model import build_model
from makers.utils import *

from keras import Model
from keras.layers import Input, LSTM, Conv3D, Reshape, Dense, BatchNormalization, Dropout, concatenate, MaxPooling3D, Lambda, Layer, Activation
from tensorflow.keras.optimizers import Adam

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

class ConvBook(Layer):

    def __init__(self, total_depth, n_hidden_units, dropout_rate=0.5, **kwargs):
        super().__init__(**kwargs)

        self.layers = [

            # conv across price+volume
            Conv3D(n_hidden_units, kernel_size=(1, 1, 2), strides=(1, 1, 2), activation='relu'),
            Conv3D(n_hidden_units, kernel_size=(4, 1, 1), activation='relu', padding='same'),
            Conv3D(n_hidden_units, kernel_size=(4, 1, 1), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(dropout_rate),

            # conv across bid+ask
            Conv3D(n_hidden_units, kernel_size=(1, 1, 2), strides=(1, 1, 2), activation='relu'),
            Conv3D(n_hidden_units, kernel_size=(4, 1, 1), activation='relu', padding='same'),
            Conv3D(n_hidden_units, kernel_size=(4, 1, 1), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(dropout_rate),

            # conv across price levels
            Conv3D(n_hidden_units, kernel_size=(1, 1, total_depth), activation='relu'),
            Conv3D(n_hidden_units, kernel_size=(4, 1, 1), activation='relu', padding='same'),
            Conv3D(n_hidden_units, kernel_size=(4, 1, 1), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Dropout(dropout_rate),

        ]

    def call(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class Inception(Layer):

    def __init__(self, n_hidden_units, dropout_rate=0.5, **kwargs):
        super().__init__(**kwargs)

        self.l11 = Conv3D(n_hidden_units, kernel_size=(1, 1, 1), activation='relu', padding='same')
        self.l12 = Conv3D(n_hidden_units, kernel_size=(3, 1, 1), padding='same')
        self.l13 = BatchNormalization()
        self.l14 = Activation('relu')
        self.l15 = Dropout(dropout_rate)

        self.l21 = Conv3D(n_hidden_units, kernel_size=(1, 1, 1), activation='relu', padding='same')
        self.l22 = Conv3D(n_hidden_units, kernel_size=(5, 1, 1), padding='same')
        self.l23 = BatchNormalization()
        self.l24 = Activation('relu')
        self.l25 = Dropout(dropout_rate)

        self.l31 = MaxPooling3D(pool_size=(3, 1, 1), strides=(1, 1, 1), padding='same')
        self.l32 = Conv3D(n_hidden_units, kernel_size=(1, 1, 1), padding='same')
        self.l33 = BatchNormalization()
        self.l34 = Activation('relu')
        self.l35 = Dropout(dropout_rate)

    def call(self, x):
        y1 = self.l11(x)
        y1 = self.l12(y1)
        y1 = self.l13(y1)
        y1 = self.l14(y1)
        y1 = self.l15(y1)

        y2 = self.l21(x)
        y2 = self.l22(y2)
        y2 = self.l23(y2)
        y2 = self.l24(y2)
        y2 = self.l25(y2)

        y3 = self.l31(x)
        y3 = self.l32(y3)
        y3 = self.l33(y3)
        y3 = self.l34(y3)
        y3 = self.l35(y3)

        return concatenate([y1, y2, y3])


class ConvFeeds(Layer):

    def __init__(self, window_size, n_targets, n_assets, n_exchanges, n_hidden_units, **kwargs):
        super().__init__(**kwargs)

        self.layers = []

        # conv across assets
        self.layers += [
            Conv3D(n_hidden_units * n_targets, kernel_size=(1, n_assets, 1), strides=(1, n_assets, 1),
                   activation='relu'),
        ]

        # conv across exchanges
        self.layers += [
            Conv3D(n_hidden_units * n_targets, kernel_size=(1, n_exchanges, 1), strides=(1, n_exchanges, 1),
                   activation='relu'),
        ]

    def call(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


# requires downgrade to numpy=1.19
class Recurrent(Layer):

    def __init__(self, n_hidden_units, dropout_rate, **kwargs):
        super().__init__(**kwargs)

        self.layers = [
            LSTM(n_hidden_units),
            BatchNormalization(),
            Dropout(dropout_rate),
        ]

    def call(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


def build_model(window_size, n_exchanges, n_assets, n_targets, total_depth):
    inputs = Input(shape=(window_size, n_exchanges * n_assets, total_depth * 4, 1))

    x = ConvBook(total_depth=total_depth, n_hidden_units=64)(inputs)
    x = Inception(n_hidden_units=64)(x)
    # x = ConvFeeds(n_hidden_units=16, window_size=window_size, n_targets=n_targets, n_exchanges=n_exchanges, n_assets=n_assets)(x)
    x = Reshape((window_size, n_targets, -1))(x)

    # unstack
    unstacked = Lambda(lambda x: tf.unstack(x, axis=2))(x)

    rec = Recurrent(n_hidden_units=64, dropout_rate=0.8)
    outputs = [rec(x) for x in unstacked]

    # fully-connected softmax output
    layer = Dense(3, activation='softmax')
    outputs = [layer(x) for x in outputs]

    adam = Adam(learning_rate=1e-4)

    model = Model(inputs, outputs)
    model.compile(optimizer=adam, loss='categorical_crossentropy')

    return model


def main():
    # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    CONFIG_PATH = "_input/config.yml"
    OUTDIR = '_output'
    MODEL_TAG = 'Multi-model basic'

    # load config
    config = yaml.safe_load(open(CONFIG_PATH, 'r'))

    # specify paths
    model_name = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(OUTDIR, 'models', model_name)
    model_path = os.path.join(model_dir, 'model.h5')
    metrics_path = os.path.join(model_dir, 'metrics.txt')
    config_path = os.path.join(model_dir, 'config.yml')

    os.makedirs(model_dir, exist_ok=True)

    lob_depth = 1
    window_size = 20
    n_targets = 1
    n_exchanges = 1
    n_assets = 1

    # load custom
    df_total = pd.read_csv('_input/COINBASE-BTC-USD-book-trades.csv')
    # df_total.loc[df_total.bv == 0., 'bp'] = df_total.ap0
    # df_total.loc[df_total.sv == 0., 'sp'] = df_total.bp0
    books = [df_total]
    feeds = [f'COINBASE-BTC-USD']

    # calculate smoothed labels
    # https://arxiv.org/pdf/1808.03668.pdf, p. 4

    # use 10, given 100ms interval corresponds to 1s
    kernel_size = 20
    label_threshold = 2e-5

    for df in books:
        # midprice
        mid_price = (df['ap0'] + df['bp0']) / 2

        df['m+'] = mid_price.shift(-(kernel_size - 1)).rolling(kernel_size).mean()
        df['m-'] = mid_price.rolling(kernel_size).mean()
        df['y2'] = (df['m+'] - df['m-']) / df['m-']

        # signal3 might be less useful since it might indicate jump in the past, not future
        df['signal2'] = -1 + 1. * (df['y2'] >= -label_threshold) + 1. * (df['y2'] >= label_threshold)

    df_targets = pd.concat([df['signal2'].rename(f'target{i}') for i, df in enumerate(books)], axis=1)

    features = ['COINBASE-BTC-USD']
    targets = ['COINBASE-BTC-USD']

    def prepare_trades(books, targets, scaler, lob_depth, n_feeds, window_size):
        columns = get_lob_columns(lob_depth) + ['bp', 'bv', 'sp', 'sv']
        X = np.stack([books[i][columns].to_numpy() for i in range(len(books))]).transpose(1, 0, 2)

        # create windows
        idxnum = range(window_size, X.shape[0])
        idx = [np.arange(x - window_size + 1, x + 1) for x in idxnum]
        X = X[idx]

        # detect price columns
        price_mask = [column[1] == 'p' for column in columns]
        price_columns = [column for column in columns if column[1] == 'p']
        print(f"Using price columns {price_columns}")

        # subtract midprice
        bid, ask = X[:, -1, :, [0, 2]]
        mid_price = (bid + ask) / 2
        X[:, :, :, price_mask] -= mid_price[:, np.newaxis, :, np.newaxis]

        # targets
        enc = OneHotEncoder(sparse=False)
        y = np.array([enc.fit_transform(target.reshape(-1, 1)) for target in targets.to_numpy()[idxnum].T]).transpose(1,
                                                                                                                      0,
                                                                                                                      2)

        # split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # scale training set
        n_samples = X_train.shape[0]
        X_train[:] = scaler.fit_transform(X_train.reshape((n_samples, -1))).reshape(
            (n_samples, window_size, n_feeds, 4 * (lob_depth + 1)))
        X_train = X_train[:, :, :, :, np.newaxis]

        # scale test set
        n_samples = X_test.shape[0]
        X_test[:] = scaler.transform(X_test.reshape((n_samples, -1))).reshape(
            (n_samples, window_size, n_feeds, 4 * (lob_depth + 1)))
        X_test = X_test[:, :, :, :, np.newaxis]

        return X_train, X_test, y_train, y_test

    books_used = [book for x, book in zip([feed in features for feed in feeds], books) if x]
    targets_used = df_targets.loc[:, [feed in targets for feed in feeds]]

    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = prepare_trades(books_used, targets_used, scaler, lob_depth,
                                                      n_exchanges * n_assets, window_size)

    # dump scaler into file
    scaler_path = os.path.join(model_dir, 'scaler.h5')
    joblib.dump(scaler, scaler_path)


    # build and train model
    # takes depth+1 since we also supply MO data
    model = build_model(window_size=config['model']['window_size'],
                        n_exchanges=n_exchanges,
                        n_assets=n_assets,
                        n_targets=n_targets,
                        total_depth=lob_depth + 1)

    cp_callback = ModelCheckpoint(filepath=model_path, monitor='val_loss', save_weights_only=True, save_best_only=True)

    history = model.fit(X_train, tf.unstack(y_train, axis=1),
                        epochs=40,
                        batch_size=128,
                        validation_split=0.15, callbacks=[cp_callback],
                        verbose=2)

    # save history
    history_path = os.path.join(model_dir, 'history.csv')
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(history_path, index=False)

    # load the best model
    model.load_weights(model_path)

    # training performance

    def get_perf(X, y, label):

        loss = model.evaluate(X, tf.unstack(y, axis=1), verbose=0)

        if n_targets == 1:
            loss = [loss]

        y_pred = model.predict(X)

        if n_targets == 1:
            y_pred = np.expand_dims(y_pred, 0)

        y_pred_num = np.argmax(y_pred, axis=2)
        y_test_num = np.argmax(y, axis=2).reshape(-1, n_targets).transpose()

        metrics = ""

        for i, target in enumerate(targets):
            metrics += f"{target}:\n"
            metrics += get_metrics(y_test_num[i], y_pred_num[i])
            metrics += "\n"

        loss_string = '\n'.join(['{:7.4}'.format(l) for l in loss])

        output = (
            f"{label}:\n\n"
            f"{'losses:':20}\n{loss_string}\n\n"
            f"{metrics}"
        )

        return output

    output_training = get_perf(X_train, y_train, 'training')
    output_testing = get_perf(X_test, y_test, 'testing')

    output = f"# {MODEL_TAG}\n\n"
    output += '\n'.join([x + y for x, y in zip([x.ljust(48) for x in output_training.split('\n')],
                                               [x.ljust(48) for x in output_testing.split('\n')])])
    print(output)

    f = open(metrics_path, 'w')
    f.write(output)
    f.close()
    print(f"Metrics written to {metrics_path}.")

    yaml.dump(config, open(config_path, 'w'))
    print(f"Config copied to {config_path}.")


if __name__ == '__main__':
    main()
