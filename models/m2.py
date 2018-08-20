try:
    from tensorflow.python.keras.models import Sequential
    from tensorflow.python.keras.optimizers import Adam
    from tensorflow.python.keras.layers import Conv2D, Flatten, Dense, AlphaDropout
    from tensorflow.python.keras.layers import MaxPooling2D, AveragePooling2D
except ImportError:
    from tensorflow.contrib.keras.python.keras.models import Sequential
    from tensorflow.contrib.keras.python.keras.optimizers import Adam
    from tensorflow.contrib.keras.python.keras.layers import Conv2D, Flatten, Dense, AlphaDropout
    from tensorflow.contrib.keras.python.keras.layers.pooling import MaxPooling2D, AveragePooling2D

preprocess_config = {
    'shift_range': 5,
    'rotate_range': 360.0,
    'flip_ud': True,
    'flip_lr': True,
    'scale_range': 1.25,
    'pca_whitening': False,
    'pca_epsilon': 0.1
}

train_config = {
    'batch_size': 64
}

def get_model(input_shape):

    model = Sequential()

    model.add(Conv2D(16, 5, padding='same', kernel_initializer='lecun_normal', activation='selu', input_shape=input_shape))
    model.add(Conv2D(16, 5, padding='same', kernel_initializer='lecun_normal', activation='selu'))
    model.add(Conv2D(16, 5, padding='same', kernel_initializer='lecun_normal', activation='selu'))
    model.add(Conv2D(16, 5, padding='same', kernel_initializer='lecun_normal', activation='selu'))
    model.add(Conv2D(16, 5, padding='same', kernel_initializer='lecun_normal', activation='selu'))
    model.add(Conv2D(16, 5, padding='same', kernel_initializer='lecun_normal', activation='selu'))
    model.add(Conv2D(16, 5, padding='same', kernel_initializer='lecun_normal', activation='selu'))
    model.add(Conv2D(16, 5, padding='same', kernel_initializer='lecun_normal', activation='selu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(32, 3, padding='same', kernel_initializer='lecun_normal', activation='selu'))
    model.add(Conv2D(32, 3, padding='same', kernel_initializer='lecun_normal', activation='selu'))
    model.add(Conv2D(32, 3, padding='same', kernel_initializer='lecun_normal', activation='selu'))
    model.add(Conv2D(32, 3, padding='same', kernel_initializer='lecun_normal', activation='selu'))
    model.add(Conv2D(32, 3, padding='same', kernel_initializer='lecun_normal', activation='selu'))
    model.add(Conv2D(32, 3, padding='same', kernel_initializer='lecun_normal', activation='selu'))
    model.add(Conv2D(32, 3, padding='same', kernel_initializer='lecun_normal', activation='selu'))
    model.add(Conv2D(32, 3, padding='same', kernel_initializer='lecun_normal', activation='selu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(64, 3, padding='same', kernel_initializer='lecun_normal', activation='selu'))
    model.add(Conv2D(64, 3, padding='same', kernel_initializer='lecun_normal', activation='selu'))
    model.add(Conv2D(64, 3, padding='same', kernel_initializer='lecun_normal', activation='selu'))
    model.add(Conv2D(64, 3, padding='same', kernel_initializer='lecun_normal', activation='selu'))
    model.add(Conv2D(64, 3, padding='same', kernel_initializer='lecun_normal', activation='selu'))
    model.add(Conv2D(64, 3, padding='same', kernel_initializer='lecun_normal', activation='selu'))
    model.add(Conv2D(64, 3, padding='same', kernel_initializer='lecun_normal', activation='selu'))
    model.add(Conv2D(64, 3, padding='same', kernel_initializer='lecun_normal', activation='selu'))
    model.add(AveragePooling2D())

    model.add(Flatten())

    model.add(Dense(1024, kernel_initializer='lecun_normal', activation='selu'))
    model.add(AlphaDropout(0.2))
    model.add(Dense(1024, kernel_initializer='lecun_normal', activation='selu'))
    model.add(AlphaDropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['binary_accuracy'])

    return model
