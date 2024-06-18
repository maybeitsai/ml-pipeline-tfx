import kerastuner as kt
import tensorflow as tf
from tensorflow.keras import layers # type: ignore
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.tuner.component import TunerFnResult
import tensorflow_transform as tft

LABEL_KEY = "Label"
FEATURE_KEY = "Sentence"

def transformed_name(key):
    return key + "_xf"

def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

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

def build_keras_model(hp, vectorize_layer):
    inputs = layers.Input(shape=(1,), dtype=tf.string, name=transformed_name(FEATURE_KEY))
    reshaped_narrative = tf.reshape(inputs, [-1])
    x = vectorize_layer(reshaped_narrative)
    x = layers.Embedding(input_dim=vectorize_layer.vocabulary_size(), output_dim=hp.Int('embedding_dim', min_value=16, max_value=128, step=16))(x)
    x = layers.Conv1D(filters=hp.Int('conv1d', min_value=32, max_value=128, step=16), kernel_size=3, activation='relu')(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(units=hp.Int('fc', min_value=32, max_value=256, step=32), activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def tuner_fn(fn_args: FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_set = input_fn(fn_args.train_files, tf_transform_output, num_epochs=5)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, num_epochs=5)

    def model_builder(hp):
        # Create and adapt the vectorize_layer
        vectorize_layer = layers.TextVectorization(
            max_tokens=hp.Int('vocab_size', min_value=8000, max_value=15000, step=1000),
            output_mode='int',
            output_sequence_length=hp.Int('sequence_length', min_value=50, max_value=150, step=25)
        )
        
        raw_train_dataset = train_set.map(lambda x, y: x[transformed_name(FEATURE_KEY)])
        vectorize_layer.adapt(raw_train_dataset)

        return build_keras_model(hp, vectorize_layer)

    tuner = kt.RandomSearch(
        model_builder,
        objective='val_accuracy',
        seed=42,
        max_trials=50,
        directory=fn_args.working_dir,
        project_name='xss_tuning')

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_set,
            "validation_data": val_set,
            "epochs": 5
        }
    )
