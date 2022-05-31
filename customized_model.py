import tensorflow as tf


Pretrain = tf.keras.applications.MobileNetV2
pretrain_model = Pretrain(input_shape=(224, 224, 3), include_top=False, pooling='avg')
for layer in pretrain_model.layers:
    layer.trainable = False


def style_classifier(model):
    fc0 = tf.keras.layers.Dense(512, activation='sigmoid')(model.outputs[0])
    dropout0 = tf.keras.layers.Dropout(0.3)(fc0)
    fc1 = tf.keras.layers.Dense(256, activation='sigmoid')(dropout0)
    dropout1 = tf.keras.layers.Dropout(0.3)(fc1)
    fc2 = tf.keras.layers.Dense(50, activation='softmax')(dropout1)
    return tf.keras.Model(inputs=model.inputs, outputs=fc2)


def get_model():
    classifier = style_classifier(pretrain_model)
    classifier.load_weights("custom.hdf5")
    return classifier