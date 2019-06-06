from keras import Sequential
from keras.layers import Conv2D, Dense, Flatten
from keras.optimizers import Adam

class DQN_Model:
    def __init__(self, action_size, state_size):
        self.action_size = action_size
        self.state_size = state_size
        self.learning_rate = 0.001
        self.model = self._build_model()
        
    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(
            32,
            (8, 8),
            strides = 4,
            activation = 'relu',
            input_shape = self.state_size
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
        model.add(Dense(self.action_size, activation = 'softmax'))

        model.compile(
            optimizer = Adam(lr = self.learning_rate),
            loss = 'mse',
            metrics=['accuracy']
        )
        return model
