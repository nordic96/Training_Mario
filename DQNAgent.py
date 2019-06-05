from keras import Sequential
from keras.layers import Conv2D, Dense, Flatten
from keras.optimizers import Adam

def _build_model(action_size, state_size):
    model = Sequential()
    model.add(Conv2D(
        32,
        (8, 8),
        strides = 4,
        activation = 'relu',
        input_shape = state_size
    ))
    model.add(Conv2D(
        64,
        (4, 4),
        strides = 4,
        activation = 'relu',
    ))
    model.add(Conv2D(
        64,
        (3, 3),
        strides = 1,
        activation = 'relu',
    ))
     
    model.add(Flatten())
    model.add(Dense(512, activation='relu')) #hidden
    model.add(Dense(action_size, activation = 'softmax'))

    model.compile(
        optimizer = Adam(lr = 0.001),
        loss = 'mse',
        metrics=['accuracy']
    )
    return model
