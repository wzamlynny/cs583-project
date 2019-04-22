from keras.models import Sequential, Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint

class KaggleModel:
    def __init__(self, model, train, test):
        model.summary()
        self.model = model
        self.train_data = train
        self.test_data = test

    def compile(self):
        """ Compiles the model. Should be defined by the user.
        """
        raise NotImplementedError()

    def train(self, epochs=1):
        """ Default training behavior. Simply does a Model.fit(X, Y).
        """
        # Get the training data
        X_train, Y_train = self.train_data

        checkpoint = ModelCheckpoint('model.h5')
        
        # Fit to the data
        self.model.fit(X_train, Y_train, epochs=epochs, validation_data=self.test_data, callbacks=[checkpoint])

    def predict(self, X):
        return self.model.predict(X)


def ResidualBlock(mdl):
    x = Input(shape=mdl.input_shape[1:])

    y = mdl(x)
    y = Add()([x, y])

    return Model(x, y)

class ImageFreeModel(KaggleModel):
    def __init__(self, train, test):
        kernel = Sequential(name='image_free_encoder')
        
        kernel.add(BatchNormalization(input_shape=(25,)))
        
        kernel.add(Dense(128))
        kernel.add(Activation('relu'))
        
        # Use a single dense residual block
        """
        blk = Sequential()
        
        blk.add(Dropout(0.5, input_shape=kernel.output_shape[1:]))
        blk.add(Dense(128))
        blk.add(Activation('relu'))

        blk.add(Dense(128))
        blk.add(Activation('relu'))

        kernel.add(ResidualBlock(blk))
        """

        model = Sequential(name='image_free')
        model.add(kernel)
        
        model.add(Dense(64, activation='relu'))

        # Labels are one of [0, 1, 2, 3, 4]
        model.add(Dense(5, activation='softmax'))

        # Build using the built model
        super().__init__(model, train, test)

    def compile(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

class SingleImageModel(KaggleModel):
    def __init__(self, train, test):
        # The model takes in attributes and an image.
        kernel = Sequential(name='single_image_encoder')
        
        # Architecture for the images
        f = 64
        kernel.add(Conv2D(f, kernel_size=(7,7), strides=(1,1), padding='same', input_shape=(64, 64, 3)))
        kernel.add(BatchNormalization())

        while kernel.output_shape[1] > 4:
            for _ in range(2):
                # Old output is kept as residue
                x = z = Input(shape=(kernel.output_shape[1:]))

                z = Activation('relu')(z)
                z = Dropout(0.5)(z)
                z = Conv2D(f, kernel_size=(5,5), strides=(1,1), padding='same')(z)
                z = BatchNormalization()(z)

                z = Activation('relu')(z)
                z = Dropout(0.5)(z)
                z = Conv2D(f, kernel_size=(5,5), strides=(1,1), padding='same')(z)
                z = BatchNormalization()(z)
                
                # The sum of the residue and the new computation
                y = Add()([x, z])
                
                # Add the residual block
                blk = Model(x, y)
                kernel.add(blk)
            
            # Reduce dimension
            kernel.add(MaxPooling2D((2,2)))
            kernel.add(Dropout(0.5))
            kernel.add(Conv2D(2*f, kernel_size=(1,1), strides=(1,1), padding='same', input_shape=(64, 64, 3)))
            f *= 2
        
        
        kernel.add(Flatten())
        
        kernel.add(Dropout(0.5))
        kernel.add(Dense(128, activation='relu'))
        
        kernel.summary()

        model = Sequential(name='single_image')

        model.add(kernel)
        model.add(Dense(5, activation='softmax'))

        super().__init__(model, train, test)

    def compile(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

class TextOnlyModel(KaggleModel):
    def __init__(self, train, test, vocabulary, sequence_length_w=30):
        embedding_dim = 32

        # The model takes in attributes and an image.
        kernel = Sequential(name='description_encoder')
        
        # Architecture for the images
        kernel.add(Embedding(vocabulary, embedding_dim, input_length=sequence_length_w))

        kernel.add(LSTM(embedding_dim, return_sequences=True, dropout=0.2))
        kernel.add(LSTM(embedding_dim, return_sequences=True, dropout=0.2))
        kernel.add(LSTM(embedding_dim, return_sequences=False, dropout=0.2))

        kernel.add(Dense(32, activation='relu'))

        model = Sequential(name='description')
        model.add(kernel)
        model.add(Dense(5, activation='softmax'))

        super().__init__(model, train, test)

    def compile(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

class UnionModel(KaggleModel):
    def __init__(self, models, train, test, freeze=True):
        
        xs = []
        ys = []
        for model in models:
            model = model.model
            x = Input(shape=model.input_shape[1:], name='{}_in'.format(model.name))
            # Get the first layer of the model. This is the encoder
            layer = model.get_layer(index=0)
            
            # It must not be trainable
            if freeze: layer.trainable = False
            # The output only utilizes the encoder component
            y = layer(x)
            
            # Output should be flat
            if len(y.shape) > 2:
                y = Flatten()(y)
                
            # Save values
            xs.append(x)
            ys.append(y)

        y = Concatenate()(ys)

        y = Dense(128)(y)
        y = BatchNormalization()(y)

        y = Dense(5, activation='softmax')(y)
        
        model = Model(xs, y)

        super().__init__(model, train, test)

    def compile(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

