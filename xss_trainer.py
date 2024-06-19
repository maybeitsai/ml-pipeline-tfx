# File ini berisi fungsi-fungsi untuk melatih model menggunakan data yang sudah ditransformasi.

# Import Library
import tensorflow as tf
import os
import tensorflow_transform as tft
from tensorflow.keras import layers # type: ignore
from tfx.components.trainer.fn_args_utils import FnArgs

# Variabel Global
LABEL_KEY = "Label"
FEATURE_KEY = "Sentence"

# Renaming transformed features
# Menambahkan suffix _xf ke kunci fitur untuk menunjukkan bahwa fitur tersebut telah ditransformasi.
def transformed_name(key):
    return key + "_xf"

# Membaca dataset yang dikompres dengan GZIP.
def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

# Membuat dataset yang dibatch menggunakan spesifikasi fitur yang sudah ditransformasi.
def input_fn(file_pattern, tf_transform_output, num_epochs=None, batch_size=32):
    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY))
    return dataset

# Membangun dan mengkompilasi model Keras dengan lapisan-lapisan yang ditentukan oleh hyperparameter yang telah diparsing.
def build_keras_model(hp, vectorize_layer):
    inputs = layers.Input(shape=(1,), dtype=tf.string, name=transformed_name(FEATURE_KEY))
    reshaped_narrative = tf.reshape(inputs, [-1])
    x = vectorize_layer(reshaped_narrative)
    x = layers.Embedding(input_dim=vectorize_layer.vocabulary_size(), output_dim=hp['embedding_dim'])(x)
    x = layers.Conv1D(filters=hp['conv1d'], kernel_size=3, activation='relu')(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(units=hp['fc'], activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(hp['learning_rate']),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    return model

# Mendefinisikan fungsi untuk serving untuk memuat model Keras.
def _get_serve_tf_examples_fn(model, tf_transform_output):
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function # contoh TensorFlow yang di-serialize.
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)

    return serve_tf_examples_fn

# Fungsi untuk melatih model Keras dengan data yang ditransformasi dan hyperparameter terbaik dari Tuner.
def run_fn(fn_args: FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_set = input_fn(fn_args.train_files, tf_transform_output, num_epochs=10)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, num_epochs=10)

    # Parse hyperparameters directly
    hyperparameters = fn_args.hyperparameters
    raw_train_dataset = train_set.map(lambda x, y: x[transformed_name(FEATURE_KEY)])
    vectorize_layer = layers.TextVectorization(
        max_tokens=hyperparameters['values']['vocab_size'],
        output_mode='int',
        output_sequence_length=hyperparameters['values']['sequence_length'])

    vectorize_layer.adapt(raw_train_dataset)

    model = build_keras_model(hyperparameters['values'], vectorize_layer)

    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='batch')
    es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=3)
    mc = tf.keras.callbacks.ModelCheckpoint(fn_args.serving_model_dir, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

    model.fit(
        train_set,
        validation_data=val_set,
        callbacks=[tensorboard_callback, es, mc],
        epochs=10)

    signatures = {
        'serving_default': _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name='examples'))
    }
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)