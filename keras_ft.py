#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#http://marubon-ds.blogspot.com.br/2017/08/how-to-make-fine-tuning-model.html
#load CNN top=false
#predict
#train small dense NN with predictions
#
#load CNN top=false
#put small dense NN on top
#freeze all layers except the last conv block and the dense part
#train with a very small learning rate

from datetime import datetime
import keras
from keras import applications, optimizers
from keras.models import Model
from keras.layers import Input, Dense, Flatten, LeakyReLU, Activation, GlobalAveragePooling2D
from keras.initializers import Constant
from keras.callbacks import EarlyStopping, ModelCheckpoint

class KerasFineTuner():
    def __init__(self, model_name, input_shape, n_classes, weights='imagenet', model=None):
        if model is not None:
            self.model = model
        else:
            self.models = dict()
            self.models['vgg16'] = applications.vgg16.VGG16
            self.models['vgg19'] = applications.vgg19.VGG19
            self.models['resnet50'] = applications.resnet50.ResNet50
            self.models['inceptionv3'] = applications.inception_v3.InceptionV3
            self.models['inception_resnet_v2'] = applications.inception_resnet_v2.InceptionResNetV2
            self.models['xception'] = applications.xception.Xception
            self.models['densenet121'] = applications.densenet.DenseNet121
            self.models['densenet169'] = applications.densenet.DenseNet169
            self.models['densenet201'] = applications.densenet.DenseNet201

            self.model_last_freeze_layer = dict()
            self.model_last_freeze_layer['vgg16'] = 15
            self.model_last_freeze_layer['vgg19'] = 17
            self.model_last_freeze_layer['resnet50'] = 154
            self.model_last_freeze_layer['inceptionv3'] = 249
            self.model_last_freeze_layer['inception_resnet_v2'] = 752
            self.model_last_freeze_layer['xception'] = 122
            self.model_last_freeze_layer['densenet121'] = 413
            self.model_last_freeze_layer['densenet169'] = 574
            self.model_last_freeze_layer['densenet201'] = 686

            model_type = self.models.get(model_name, None)
        
            if model_type is None:
                raise('Invalid model_name')

            self.model = model_type(include_top=False, weights=weights, input_shape=input_shape)
            self.last_freeze_layer = self.model_last_freeze_layer[model_name]

        ts = datetime.now()
        ts_str = datetime.strftime(ts, '%Y%m%d_%H%M%S')

        self.top_chkpt_file_name = 'top_%s_%s.h5' % (model_name, ts_str)
        self.ft_chkpt_file_name = 'ft_%s_%s.h5' % (model_name, ts_str)

        stopper = EarlyStopping(monitor='val_acc', min_delta=0.00005, patience=5, verbose=1, mode='max')
        top_chkpt = ModelCheckpoint(self.top_chkpt_file_name, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1)
        ft_chkpt = ModelCheckpoint(self.ft_chkpt_file_name, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1)

        self.top_callbacks = [stopper, top_chkpt]
        self.ft_callbacks = [stopper, ft_chkpt]

        self.n_classes = n_classes
        self.prepare_model()

    def prepare_model(self):
        for l in self.model.layers:
            l.trainable = False

        if self.n_classes == 2:
            self.loss = 'binary_crossentropy'
            self.activation = 'sigmoid'
        else:
            self.loss = 'categorical_crossentropy'
            self.activation = 'softmax'

        x = self.model.output
        x = GlobalAveragePooling2D()(x)

        self.embedder = Model(inputs=self.model.input, outputs=x)

        x = Dense(4096, kernel_initializer='glorot_uniform', bias_initializer=Constant(value=0.01))(x)
        x = LeakyReLU()(x)
        # x = Dense(4096, kernel_initializer='glorot_uniform', bias_initializer=Constant(value=0.01))(x)
        # x = LeakyReLU()(x)
        predictions = Dense(self.n_classes, activation=self.activation)(x)

        self.model = Model(inputs=self.model.input, outputs=predictions)
        self.model.compile(loss=self.loss, optimizer='adam', metrics=['accuracy'])

    def load_weights(self, weights_file):
        self.model.load_weights(weights_file)

    def get_embeddings(self, X):
        return self.embedder.predict(X)

    def fit_top_model(self, X_train, y_train, X_valid, y_valid, batch_size=16, epochs=30, datagen=None, img_mult=1):
        self.model.compile(loss=self.loss, optimizer='adam', metrics=['accuracy'])

        if datagen is None:
            self.model.fit( X_train,
                            y_train,
                            validation_data=(X_valid, y_valid),
                            batch_size=batch_size,
                            verbose=1,
                            epochs=epochs,
                            callbacks=self.top_callbacks)
        else:
            self.model.fit_generator(   datagen.flow(X_train, y_train, batch_size=batch_size, seed=42),
                                        steps_per_epoch=(len(X_train) // batch_size) * img_mult,
                                        epochs=epochs,
                                        validation_data=datagen.flow(X_valid, y_valid, batch_size=batch_size),
                                        validation_steps=(len(X_valid) // batch_size) * img_mult,
                                        verbose=1, callbacks=self.top_callbacks)

        self.model.load_weights(self.top_chkpt_file_name)

    def fine_tune_model(self, X_train, y_train, X_valid, y_valid, batch_size=16, epochs=30, datagen=None, img_mult=1):
        for l in self.model.layers[:self.last_freeze_layer]:
            l.trainable = False

        for l in self.model.layers[self.last_freeze_layer:]:
            l.trainable = True

        self.model.compile(loss=self.loss, optimizer=optimizers.SGD(lr=1e-5, momentum=0.9), metrics=['accuracy'])

        if datagen is None:
            self.model.fit( X_train,
                            y_train,
                            validation_data=(X_valid, y_valid),
                            batch_size=batch_size,
                            verbose=1,
                            epochs=epochs,
                            callbacks=self.ft_callbacks)
        else:
            self.model.fit_generator(   datagen.flow(X_train, y_train, batch_size=batch_size, seed=42),
                                        steps_per_epoch=(len(X_train) // batch_size) * img_mult,
                                        epochs=epochs,
                                        validation_data=datagen.flow(X_valid, y_valid, batch_size=batch_size),
                                        validation_steps=(len(X_valid) // batch_size) * img_mult,
                                        verbose=1,
                                        callbacks=self.ft_callbacks)

        self.model.load_weights(self.ft_chkpt_file_name)

    def predict(self, X):
        return self.model.predict(X)
