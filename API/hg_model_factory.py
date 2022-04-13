from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import SentenceTransformer

import tensorflow as tf

class HGModelFactory:
    @staticmethod
    def get_hugging_face_auto_tokenizer(m_name):
        return AutoTokenizer.from_pretrained(m_name)

    @staticmethod
    def get_hugging_face_bi_auto_model(m_name,
                                    n_labels=1,
                                    weights_path=None):
        model = TFAutoModelForSequenceClassification.from_pretrained(m_name,
                                                                     num_labels=n_labels)
        if weights_path is not None:
            model.load_weights(weights_path)
        model.layers[-1].activation = tf.keras.activations.sigmoid  # default - linear
        return model

    @staticmethod
    def get_cross_encoder(m_name):
        return CrossEncoder(m_name)

    @staticmethod
    def get_sentence_transformer(m_name):
        return SentenceTransformer(m_name)

