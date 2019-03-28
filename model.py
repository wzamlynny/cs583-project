
from keras.models import Sequential, Model
from keras.layers import *

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
        
        # Fit to the data
        self.model.fit(X_train, Y_train, epochs=epochs, validation_data=self.test_data)

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
        y_attr = Dense(32, activation='relu')(y_attr)
        y_attr = Dense(32, activation='relu')(y_attr)

        # Architecture for the images
        y_img = img

        while y_img.shape[1] > 8:
            y_img = Conv2D(64, kernel_size=(5,5), strides=(1,1), activation='relu', padding='same')(y_img)
            y_img = Conv2D(64, kernel_size=(5,5), strides=(1,1), activation='relu', padding='same')(y_img)
            y_img = MaxPooling2D(pool_size=(2,2))(y_img)
        
        y_img = AveragePooling2D(pool_size=(y_img.shape[1], y_img.shape[2]))(y_img)
        y_img = Reshape((64,))(y_img)
        y_img = Dense(1024, activation='relu')(y_img)
        y_img = Dense(32, activation='relu')(y_img)

        # Final processing
        y = Concatenate()([y_attr, y_img])
        y = Dense(32, activation='sigmoid')(y)
        y = Dense(5, activation='softmax')(y)

        model = Model([attr, img], y)
        model.name = 'conv_model'

        super().__init__(model, train, test)

    def compile(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

if __name__ == '__main__':
    mdl = ConvModel(None, None)

