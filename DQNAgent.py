import tensorflow as tf

def _build_model(action_size, state_size):
    model = tf.keras.models.Sequential([
        
        tf.keras.layers.Convolution2D(32, 8, 8, input_shape= state_size),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Convolution2D(64, 4, 4),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Convolution2D(64, 3, 3),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Dense(action_size, activation=tf.nn.softmax),

    ])

    model.compile(
        optimizer = 'adam',
        loss = 'mse',
        metrics=['accuracy']
    )
    return model
