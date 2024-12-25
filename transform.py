import tensorflow as tf

LABEL_KEY = "fraudulent"
FEATURE_KEY = "clean_text"

def transformed_name(key):
    return f"{key}_xf"

def preprocessing_fn(inputs):

    outputs = {}
    
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(inputs[FEATURE_KEY])
    
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)
    
    return outputs
