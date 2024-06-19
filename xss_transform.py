# File ini berisi fungsi-fungsi untuk memproses fitur input mentah menjadi fitur yang telah ditransformasi.

# Import Library
import tensorflow as tf

# Variabel Global
LABEL_KEY = "Label"
FEATURE_KEY = "Sentence"

# Renaming transformed features
# Menambahkan suffix _xf ke kunci fitur untuk menunjukkan bahwa fitur tersebut telah ditransformasi.
def transformed_name(key):
    return key + "_xf" 

# Preprocess input features into transformed features
# Mengubah teks menjadi huruf kecil dan mengubah tipe data label menjadi int64
def preprocessing_fn(inputs): # inputs: map from feature keys to raw features. 
    outputs = {}
    
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(inputs[FEATURE_KEY])
    
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)
    
    return outputs # outputs: map from feature keys to transformed features. 
