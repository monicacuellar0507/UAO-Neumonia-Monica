from PIL import ImageTk, Image
import tensorflow as tf

def model_fun():
    # cargar el modelo
    model_cnn = tf.keras.models.load_model('conv_MLP_84.h5')
    #model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model_cnn