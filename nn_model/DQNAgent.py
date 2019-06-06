from keras.models import Model
from keras import Sequential
from keras.layers import Input, Conv2D, Dense, Flatten, Activation, GlobalAveragePooling2D
from keras.optimizers import Adam

class DQN_Model:
    def __init__(self, action_size, state_size, model_type):
        self.action_size = action_size
        self.state_size = state_size
        self.learning_rate = 0.001
        if model_type == 'sequential':
            self.model = self._build_model()
        elif model_type == 'functional':            
            self.model = self._build_model_functional()
        
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

    def _build_model_functional(self):
        #state_size = input_shape
        input_img = Input(shape = self.state_size)
        x = Conv2D(32, (8, 8), padding='same', strides = 4)(input_img)
        x = Activation('relu')(x)

        x = Conv2D(64, (4, 4), padding='same', strides = 4)(x)
        x = Activation('relu')(x)

        x = Conv2D(64, (3, 3), padding='same', strides = 1)(x)
        x = Activation('relu')(x)
        
        #x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)
        x = Dense(512, activation='sigmoid')(x)
        
        #output
        predictions = Dense(self.action_size, activation='softmax')(x)
        model = Model(inputs = input_img, outputs = predictions)

        #compilation
        model.compile(
            optimizer = Adam(lr = self.learning_rate),
            loss = 'mean_squared_error',
            metrics = ['accuracy']
        )
                      
        return model
