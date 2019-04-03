
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

class ImageFreeModel(KaggleModel):
    def __init__(self, train, test):
        model = Sequential(name='image_free')
        
        model.add(Dense(32, activation='sigmoid', input_shape=(32,)))

        # Labels are one of [0, 1, 2, 3, 4]
        model.add(Dense(5, activation='softmax'))
        
        # Build using the built model
        super().__init__(model, train, test)

    def compile(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

class ConvModel(KaggleModel):
    def __init__(self, train, test):
        # The model takes in attributes and an image.
        attr = Input(shape=(32,))
        img = Input(shape=(64, 64, 3))
        
        # Architecture for the attributes
        y_attr = attr
        y_attr = Dense(64, activation='relu')(y_attr)
        y_attr = Dense(128, activation='relu')(y_attr)

        # Architecture for the images
        y_img = img
        y_img = Conv2D(64, kernel_size=(7,7), strides=(1,1), padding='same')(y_img)
        y_img = BatchNormalization()(y_img)

        while y_img.shape[1] > 4:
            for _ in range(2):
                # Old output is kept as residue
                y = z = y_img

                z = Activation('relu')(z)
                z = Conv2D(64, kernel_size=(5,5), strides=(1,1), padding='same')(z)
                z = BatchNormalization()(z)

                z = Activation('relu')(z)
                z = Conv2D(64, kernel_size=(5,5), strides=(1,1), padding='same')(z)
                z = BatchNormalization()(z)
                
                # The sum of the residue and the new computation
                y_img = Add()([y, z])
            
            # Reduce dimension
            y_img = MaxPooling2D((2,2))(y_img)
        
        y_img = Flatten()(y_img)
        y_img = Dense(128, activation='relu')(y_img)

        # Final processing
        y = Concatenate()([y_attr, y_img])

        y = Dropout(0.5)(y)
        y = Dense(64, activation='relu')(y)

        y = Dense(5, activation='softmax')(y)

        model = Model([attr, img], y)
        model.name = 'conv_model'

        super().__init__(model, train, test)

    def compile(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

if __name__ == '__main__':
    mdl = ConvModel(None, None)

