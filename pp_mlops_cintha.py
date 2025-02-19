# -*- coding: utf-8 -*-
"""PP_MLOps_Cintha.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/14bLAKbIz-ZLGTIw6RoyXvVtxR5Y51B2i

Nama : Cintha Hafrida Putri
ID Dicoding : cintha_bang

# Proyek Pengembangan Machine Learning Pipeline

# Import Library

Mengunduh dan menginstal TFX versi 1.15.1 beserta semua dependensinya, sehingga dapat mulai menggunakannya dalam proyek pengembangan machine learning pipeline ini.
"""

pip install tfx==1.15.1

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import nltk
import string
import tensorflow_data_validation as tfdv

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from string import punctuation

from tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Trainer, Tuner
from tfx.proto import example_gen_pb2
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')

"""# Directory Pipeline

Dalam kode ini, saya mendefinisikan beberapa variabel kunci untuk pipeline machine learning saya, termasuk PIPELINE_NAME yang menyimpan nama pipeline ("cintha_bang-pipeline"), PIPELINE_ROOT yang merupakan direktori utama untuk pipeline, SCHEMA_PIPELINE_NAME untuk nama skema validasi data ("fake-news-tfdv-schema"), METADATA_PATH yang menyimpan lokasi file metadata untuk pipeline di database SQLite, dan SERVING_MODEL_DIR yang menunjukkan direktori tempat model yang telah dilatih akan disajikan. Kode ini membantu dalam pengelolaan dan strukturisasi berbagai komponen pipeline dengan lebih baik.
"""

PIPELINE_NAME = "cintha_bang-pipeline"
PIPELINE_ROOT = os.path.join('pipelines', PIPELINE_NAME)
SCHEMA_PIPELINE_NAME = "fake-news-tfdv-schema"
METADATA_PATH = os.path.join('metadata', PIPELINE_NAME, 'metadata.db')
SERVING_MODEL_DIR = os.path.join('serving_model_dir', PIPELINE_NAME)

"""# Data Understanding

Pada tahap ini dilakukan untuk memahami dataset sebelum analisis.

## Data Loading

data_dir untuk menyimpan direktori sumber data mentah ('data'), data_clean untuk menyimpan direktori tempat data yang telah dibersihkan ('data_clean'), dan file_name untuk menyimpan nama file data mentah yang akan diproses ('fake_job_postings.csv'). Variabel-variabel ini membantu dalam pengorganisasian struktur penyimpanan data agar lebih mudah diakses selama proses pengolahan.
"""

data_dir = 'data'
data_clean = 'data_clean'
file_name = 'fake_job_postings.csv'

"""Tahap ini adalah memuat dataset. Saya menggunakan google drive agar lebih mudah.

**Dataset :** Real/Fake Job Posting Prediction dari Kaggle https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
"""

df=pd.read_csv('/content/drive/MyDrive/DICODING MACHINE LEARNING/Machine Learning Operations (MLOps)/fake_job_postings.csv')
df.head()

"""# EDA"""

df.info()

df.shape

df.describe()

df.isna().sum()

"""Karena kita akan menggabungkan kolom menjadi text maka missing value akan diubah menjadi spasi

## Data Preparation
"""

df.fillna(" ",inplace = True)

"""Menghapus kolom yang tidak diperlukan"""

df = df.drop(columns=['job_id', 'telecommuting', 'has_questions', 'has_company_logo', 'salary_range'])

df.head()

"""Membuat kolom yang berisikan text dimana text berasal dari semua kolom dataset yang disatukan"""

df['text'] = df['title'] + ' ' + df['location'] + ' ' + df['department'] + ' ' + df['company_profile'] + ' ' + df['description'] + ' ' + df['requirements'] + ' ' + df['benefits'] + ' ' + df['employment_type'] + ' ' + df['required_education'] + ' ' + df['industry'] + ' ' + df['function']

"""Cek apakah kolom text sudah masuk"""

df.head()

"""Setelah membuat kolom text maka langkah selanjutnya adalah menghapus kolom lainnya karena tidak diperlukan"""

df = df.drop(columns=['title', 'location', 'department', 'company_profile', 'description', 'requirements', 'benefits', 'employment_type', 'required_experience', 'required_education', 'industry', 'function'])

df.head()

df['tokens'] = df['text'].apply(word_tokenize)

# Inisialisasi daftar stopwords dalam bahasa Inggris
stop = set(stopwords.words('english'))

# Menghapus stopwords
df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word.lower() not in stop])

# Menggabungkan kembali token menjadi satu string dan menyimpan sebagai clean_text
df['clean_text'] = df['tokens'].apply(lambda x: ' '.join(x))

"""Langkah selanjutnya adalah mengonversi tag part-of-speech (POS) yang diperoleh dari pemrosesan teks menggunakan pustaka Natural Language Toolkit (nltk) ke dalam bentuk yang lebih sederhana, yaitu kategori POS yang dapat digunakan oleh WordNet."""

def get_simple_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

"""Melakukan proses lemmatization pada teks input dengan menghilangkan stopwords dan mengonversi kata-kata menjadi bentuk dasar atau lemma-nya."""

lemmatizer = WordNetLemmatizer()

def lemmatize_words(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            pos = pos_tag([i.strip()])
            word = lemmatizer.lemmatize(i.strip(),get_simple_pos(pos[0][1]))
            final_text.append(word.lower())
    return " ".join(final_text)

df.clean_text = df.clean_text.apply(lemmatize_words)

df.info()

"""Hapus kolom yang sudah tidak diperlukan"""

df = df.drop(columns=['tokens', 'text'])

df.head()

if not os.path.exists(data_clean):
    os.makedirs(data_clean)

df.to_csv(os.path.join(data_clean, "fake_job_postings_clean.csv"), index=False)

interactive_context = InteractiveContext(pipeline_root=PIPELINE_ROOT)

"""# Load Datset menggunakan Example Gen

Mengonfigurasi dan menginisialisasi komponen ExampleGen, yang bertanggung jawab untuk menghasilkan contoh data dari file CSV dan membagi data tersebut ke dalam beberapa set (split) untuk keperluan pelatihan dan evaluasi model.
"""

output = example_gen_pb2.Output(
    split_config=example_gen_pb2.SplitConfig(splits=[
        example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
        example_gen_pb2.SplitConfig.Split(name="test", hash_buckets=2),
    ])
)

example_gen = CsvExampleGen(input_base=data_clean, output_config=output)

interactive_context.run(example_gen)

"""## Summary Statistic

Menjalankan komponen StatisticsGen untuk menghasilkan statistik dari data contoh (examples) yang dihasilkan oleh example_gen, dalam konteks interaktif.
"""

statistics_gen = StatisticsGen(
    examples=example_gen.outputs["examples"]
)
interactive_context.run(statistics_gen)

"""Menampilkan statistik data yang dihasilkan oleh komponen StatisticsGen secara interaktif, sehingga Anda bisa melihat distribusi dan ringkasan data yang telah diproses."""

interactive_context.show(statistics_gen.outputs["statistics"])

"""# Data Schema

Membuat komponen SchemaGen yang secara otomatis menghasilkan skema (schema) data berdasarkan statistik yang dihasilkan oleh StatisticsGen.
"""

schema_gen = SchemaGen(
    statistics=statistics_gen.outputs["statistics"],
)
interactive_context.run(schema_gen)

interactive_context.show(schema_gen.outputs["schema"])

"""# Membuat Validator

Menjalankan komponen ExampleValidator, yang memvalidasi data input berdasarkan statistik dari StatisticsGen dan skema dari SchemaGen. Komponen ini mendeteksi anomali atau data yang tidak sesuai dengan skema, seperti tipe data yang salah atau nilai yang melampaui batas, sehingga memastikan kualitas dan konsistensi data dalam pipeline machine learning.
"""

example_validator = ExampleValidator(
    statistics=statistics_gen.outputs["statistics"],
    schema=schema_gen.outputs["schema"],
)
interactive_context.run(example_validator)

interactive_context.show(example_validator.outputs["anomalies"])

"""# Preprocessing Data"""

TRANSFORM_MODULE_FILE = "transform.py"

# Commented out IPython magic to ensure Python compatibility.
# %%writefile {TRANSFORM_MODULE_FILE}
# import tensorflow as tf
# 
# LABEL_KEY = "fraudulent"
# FEATURE_KEY = "clean_text"
# 
# def transformed_name(key):
#     return f"{key}_xf"
# 
# def preprocessing_fn(inputs):
# 
#     outputs = {}
# 
#     outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(inputs[FEATURE_KEY])
# 
#     outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)
# 
#     return outputs

""""overwriting transform.py" menunjukkan bahwa file tuner.py yang sudah ada sebelumnya sedang ditimpa dengan konten baru karena saya merevisi kode sehingga menjalankannya 2 kali"""

transform = Transform(
    examples=example_gen.outputs["examples"],
    schema=schema_gen.outputs["schema"],
    module_file=os.path.abspath(TRANSFORM_MODULE_FILE)
)

interactive_context.run(transform)

"""# Tuner Hyperparameter"""

TUNER_MODULE_FILE = "tuner.py"

# Commented out IPython magic to ensure Python compatibility.
# %%writefile {TUNER_MODULE_FILE}
# 
# from typing import NamedTuple, Dict, Text, Any
# import keras_tuner as kt
# import tensorflow as tf
# import tensorflow_transform as tft
# from keras_tuner.engine import base_tuner
# from keras import layers
# from tfx.components.trainer.fn_args_utils import FnArgs
# 
# LABEL_KEY = "fraudulent"
# FEATURE_KEY = "clean_text"
# NUM_EPOCHS = 5
# 
# TunerFnResult = NamedTuple("TunerFnResult", [
#     ("tuner", base_tuner.BaseTuner),
#     ("fit_kwargs", Dict[Text, Any]),
# ])
# 
# early_stopping_callback = tf.keras.callbacks.EarlyStopping(
#     monitor="val_binary_accuracy",
#     mode="max",
#     verbose=1,
#     patience=10,
# )
# 
# def transformed_name(key):
#     return f"{key}_xf"
# 
# def gzip_reader_fn(filenames):
#     return tf.data.TFRecordDataset(filenames, compression_type="GZIP")
# 
# def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=64):
#     transform_feature_spec = (
#         tf_transform_output.transformed_feature_spec().copy()
#     )
# 
#     dataset = tf.data.experimental.make_batched_features_dataset(
#         file_pattern=file_pattern,
#         batch_size=batch_size,
#         features=transform_feature_spec,
#         reader=gzip_reader_fn,
#         num_epochs=num_epochs,
#         label_key=transformed_name(LABEL_KEY),
#     )
# 
#     return dataset
# 
# def model_builder(hp, vectorizer_layer):
#     num_hidden_layers = hp.Choice(
#         "num_hidden_layers", values=[1, 2]
#     )
#     embed_dims = hp.Int(
#         "embed_dims", min_value=16, max_value=128, step=32
#     )
#     lstm_units = hp.Int(
#         "lstm_units", min_value=32, max_value=128, step=32
#     )
#     dense_units = hp.Int(
#         "dense_units", min_value=32, max_value=256, step=32
#     )
#     dropout_rate = hp.Float(
#         "dropout_rate", min_value=0.1, max_value=0.5, step=0.1
#     )
#     learning_rate = hp.Choice(
#         "learning_rate", values=[1e-2, 1e-3, 1e-4]
#     )
# 
#     inputs = tf.keras.Input(
#         shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string
#     )
# 
#     x = vectorizer_layer(inputs)
#     x = layers.Embedding(input_dim=5000, output_dim=embed_dims)(x)
#     x = layers.Bidirectional(layers.LSTM(lstm_units))(x)
# 
#     for _ in range(num_hidden_layers):
#         x = layers.Dense(dense_units, activation=tf.nn.relu)(x)
#         x = layers.Dropout(dropout_rate)(x)
# 
#     outputs = layers.Dense(1, activation=tf.nn.sigmoid)(x)
# 
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
# 
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
#         loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#         metrics=["binary_accuracy"],
#     )
# 
#     return model
# 
# 
# def tuner_fn(fn_args: FnArgs):
#     tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
# 
#     train_set = input_fn(
#         fn_args.train_files[0], tf_transform_output, NUM_EPOCHS
#     )
# 
#     eval_set = input_fn(
#         fn_args.eval_files[0], tf_transform_output, NUM_EPOCHS
#     )
# 
#     vectorizer_dataset = train_set.map(
#         lambda f, l: f[transformed_name(FEATURE_KEY)]
#     )
# 
#     vectorizer_layer = layers.TextVectorization(
#         max_tokens=5000,
#         output_mode="int",
#         output_sequence_length=500,
#     )
#     vectorizer_layer.adapt(vectorizer_dataset)
# 
#     tuner = kt.RandomSearch(
#          hypermodel=lambda hp: model_builder(hp, vectorizer_layer),
#          objective=kt.Objective('binary_accuracy', direction='max'),
#          max_trials = 5,
#          directory=fn_args.working_dir,
#          project_name="kt_RandomSearch",
#      )
# 
#     return TunerFnResult(
#         tuner=tuner,
#         fit_kwargs={
#             "callbacks": [early_stopping_callback],
#             "x": train_set,
#             "validation_data": eval_set,
#             "steps_per_epoch": fn_args.train_steps,
#             "validation_steps": fn_args.eval_steps,
#         },
#     )

""""writing tuner.py" menunjukkan bahwa kode Anda sedang disimpan ke dalam file yang bernama tuner.py

Mengkonfigurasi dan menginisialisasi komponen Tuner dalam arsitektur TFX (TensorFlow Extended)
"""

from tfx.components import Tuner
from tfx.proto import trainer_pb2

tuner = Tuner(
    module_file=os.path.abspath(TUNER_MODULE_FILE),
    examples=transform.outputs["transformed_examples"],
    transform_graph=transform.outputs["transform_graph"],
    schema=schema_gen.outputs["schema"],
    train_args=trainer_pb2.TrainArgs(splits=["train"], num_steps=20),
    eval_args=trainer_pb2.EvalArgs(splits=["test"], num_steps=5),
)

interactive_context.run(tuner)

"""Output dari pencarian hyperparameter menggunakan TFX Tuner menunjukkan bahwa percobaan ke-5 mencapai akurasi 93,2%, sementara akurasi terbaik sepanjang pencarian adalah 95,8% dari percobaan ke-3. Total waktu pencarian adalah 4 menit dan 54 detik, dengan ringkasan hasil disimpan di direktori yang ditentukan. Berikut adalah ringkasan dari percobaan terbaik:

Trial 3: Akurasi tertinggi 95,8%, hyperparameter: 1 hidden layer, 112 embedding dimensions, 32 LSTM units, 192 dense units, 0,2 dropout rate, dan learning rate 0,01.

- Trial 1: Akurasi 94,5%.
- Trial 4: Akurasi 93,2%.
- Trial 2: Akurasi 92,1%.
- Trial 0: Akurasi 91%.

# Training Model
"""

TRAINER_MODULE_FILE = "trainer.py"

# Commented out IPython magic to ensure Python compatibility.
# %%writefile {TRAINER_MODULE_FILE}
# 
# import os
# import tensorflow as tf
# import tensorflow_transform as tft
# from keras import layers
# from tfx.components.trainer.fn_args_utils import FnArgs
# 
# LABEL_KEY = "fraudulent"
# FEATURE_KEY = "clean_text"
# NUM_EPOCHS = 3
# 
# early_stopping_callback = tf.keras.callbacks.EarlyStopping(
#     monitor="val_binary_accuracy",
#     mode="max",
#     verbose=1,
#     patience=10,
# )
# 
# def transformed_name(key):
#     return key + "_xf"
# 
# 
# def gzip_reader_fn(filenames):
#     return tf.data.TFRecordDataset(filenames, compression_type='GZIP')
# 
# 
# def input_fn(file_pattern,
#              tf_transform_output,
#              num_epochs,
#              batch_size=64) -> tf.data.Dataset:
# 
#     transform_feature_spec = (
#         tf_transform_output.transformed_feature_spec().copy())
# 
#     dataset = tf.data.experimental.make_batched_features_dataset(
#         file_pattern=file_pattern,
#         batch_size=batch_size,
#         features=transform_feature_spec,
#         reader=gzip_reader_fn,
#         num_epochs=num_epochs,
#         label_key=transformed_name(LABEL_KEY))
# 
#     return dataset
# 
# def model_builder(vectorizer_layer, hyperparameters):
#     inputs = tf.keras.Input(
#         shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string
#     )
# 
#     x = vectorizer_layer(inputs)
#     x = layers.Embedding(
#         input_dim=5000,
#         output_dim=hyperparameters["embed_dims"])(x)
#     x = layers.Bidirectional(layers.LSTM(hyperparameters["lstm_units"]))(x)
# 
#     for _ in range(hyperparameters["num_hidden_layers"]):
#         x = layers.Dense(
#             hyperparameters["dense_units"],
#             activation=tf.nn.relu)(x)
#         x = layers.Dropout(hyperparameters["dropout_rate"])(x)
# 
#     outputs = layers.Dense(1, activation=tf.nn.sigmoid)(x)
# 
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
# 
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(
#             learning_rate=hyperparameters["learning_rate"]),
#         loss=tf.keras.losses.BinaryCrossentropy(),
#         metrics=[
#             tf.keras.metrics.BinaryAccuracy()],
#     )
# 
#     model.summary()
# 
#     return model
# 
# 
# def _get_serve_tf_examples_fn(model, tf_transform_output):
#     model.tft_layer = tf_transform_output.transform_features_layer()
# 
#     @tf.function
#     def serve_tf_examples_fn(serialized_tf_examples):
#         feature_spec = tf_transform_output.raw_feature_spec()
# 
#         feature_spec.pop(LABEL_KEY)
# 
#         parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
# 
#         transformed_features = model.tft_layer(parsed_features)
# 
#         return model(transformed_features)
# 
#     return serve_tf_examples_fn
# 
# 
# def run_fn(fn_args: FnArgs) -> None:
#     hyperparameters = fn_args.hyperparameters["values"]
# 
#     log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "logs")
# 
#     tensorboard_callback = tf.keras.callbacks.TensorBoard(
#         log_dir=log_dir, update_freq="batch"
#     )
# 
#     model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#         fn_args.serving_model_dir,
#         monitor="val_binary_accuracy",
#         mode="max",
#         verbose=1,
#         save_best_only=True,
#     )
# 
#     callbacks = [
#         tensorboard_callback,
#         early_stopping_callback,
#         model_checkpoint_callback
#     ]
# 
#     tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
# 
#     train_set = input_fn(
#         fn_args.train_files,
#         tf_transform_output,
#         NUM_EPOCHS)
# 
#     eval_set = input_fn(
#         fn_args.eval_files,
#         tf_transform_output,
#         NUM_EPOCHS)
# 
#     vectorizer_dataset = train_set.map(
#         lambda f, l: f[transformed_name(FEATURE_KEY)]
#     )
# 
#     vectorizer_layer = layers.TextVectorization(
#         max_tokens=5000,
#         output_mode="int",
#         output_sequence_length=500,
#     )
# 
#     vectorizer_layer.adapt(vectorizer_dataset)
# 
#     model = model_builder(vectorizer_layer, hyperparameters)
# 
#     model.fit(
#         x=train_set,
#         steps_per_epoch=fn_args.train_steps,
#         validation_data=eval_set,
#         validation_steps=fn_args.eval_steps,
#         callbacks=callbacks,
#         epochs=NUM_EPOCHS,
#         verbose=1,
#     )
# 
#     signatures = {
#         "serving_default": _get_serve_tf_examples_fn(
#             model, tf_transform_output
#         ).get_concrete_function(
#             tf.TensorSpec(
#                 shape=[None],
#                 dtype=tf.string,
#                 name="examples",
#             )
#         )
#     }
# 
#     model.save(
#         fn_args.serving_model_dir,
#         save_format="tf",
#         signatures=signatures
#     )

from tfx.proto import trainer_pb2

trainer = Trainer(
    module_file=os.path.abspath(TRAINER_MODULE_FILE),
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    hyperparameters=tuner.outputs["best_hyperparameters"],
    train_args=trainer_pb2.TrainArgs(splits=['train']),
    eval_args=trainer_pb2.EvalArgs(splits=['test'])
)

interactive_context.run(trainer)

"""**Layer Details:** Menyediakan ringkasan arsitektur model yang dibangun, termasuk jenis layer, bentuk output, dan jumlah parameter yang digunakan.

**Total Parameters:** Model memiliki total 609,793 parameter, semuanya dapat dilatih.
Epoch Information: Output menunjukkan proses pelatihan selama tiga epoch, termasuk nilai loss dan akurasi biner pada setiap epoch.

- Epoch 1: Loss 0.0525 dan akurasi 98.37% untuk data latih; akurasi validasi 98.32%.
- Epoch 2: Loss menurun menjadi 0.0090 dengan akurasi 99.74% untuk data latih; akurasi validasi meningkat menjadi 98.37%.
- Epoch 3: Loss 0.0072 dan akurasi 99.79% untuk data latih; akurasi validasi meningkat menjadi 98.40%.


Secara keseluruhan, output ini menunjukkan kemajuan pelatihan model yang efektif dengan peningkatan akurasi yang baik pada data validasi.

# Analisis dan Evaluasi Model
"""

from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import LatestBlessedModelStrategy
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing

model_resolver = Resolver(
    strategy_class=LatestBlessedModelStrategy,
    model=Channel(type=Model),
    model_blessing=Channel(type=ModelBlessing)
).with_id("latest_blessed_model_resolver")

interactive_context.run(model_resolver)

import tensorflow_model_analysis as tfma

eval_config = tfma.EvalConfig(
    model_specs=[tfma.ModelSpec(label_key='fraudulent')],
    slicing_specs=[tfma.SlicingSpec()],
    metrics_specs=[
        tfma.MetricsSpec(metrics=[
            tfma.MetricConfig(class_name='ExampleCount'),
            tfma.MetricConfig(class_name='AUC'),
            tfma.MetricConfig(class_name='FalsePositives'),
            tfma.MetricConfig(class_name='TruePositives'),
            tfma.MetricConfig(class_name='FalseNegatives'),
            tfma.MetricConfig(class_name='TrueNegatives'),
            tfma.MetricConfig(class_name='BinaryAccuracy',
                              threshold=tfma.MetricThreshold(
                                  value_threshold=tfma.GenericValueThreshold(
                                      lower_bound={'value': 0.5}),
                                  change_threshold=tfma.GenericChangeThreshold(
                                      direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                      absolute={'value': 0.0001})
                              )
            )
        ])
    ]
)

from tfx.components import Evaluator

model_analyzer = Evaluator(
    examples=example_gen.outputs["examples"],
    model=trainer.outputs["model"],
    baseline_model=model_resolver.outputs["model"],
    eval_config=eval_config,
    example_splits=['test']
)

interactive_context.run(model_analyzer)

"""memuat dan menampilkan hasil evaluasi model yang telah dilakukan menggunakan TensorFlow Model Analysis (TFMA)."""

eval_result = model_analyzer.outputs['evaluation'].get()[0].uri
tfma_result = tfma.load_eval_result(eval_result)
tfma.view.render_slicing_metrics(tfma_result)
tfma.addons.fairness.view.widget_view.render_fairness_indicator(tfma_result)

from tfx.components import Pusher
from tfx.proto import pusher_pb2

pusher = Pusher(
    model=trainer.outputs["model"],
    model_blessing=model_analyzer.outputs["blessing"],
    push_destination=pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(
            base_directory='serving_model_dir/real-or-fake-jobs-detection-model'
        )
    )
)

interactive_context.run(pusher)

!pip freeze > requirements.txt

