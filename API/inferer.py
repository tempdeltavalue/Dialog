import glob
import os

import time

from API.hg_model_factory import HGModelFactory
from API.preproc_utils import clean_text

class Result:
    def __init__(self,
                 text,
                 embs,
                 ok_preds,
                 neok_preds,
                 desc_sim_dict):

        self.id = time.time()
        self.text = text
        self.embs = embs
        self.ok_iscalm_preds = ok_preds
        self.neok_iscalm_preds = neok_preds
        self.desc_sim_dict = desc_sim_dict

class Inferer:
    def __init__(self,
                 ok_cls_model_weights_path,
                 neok_cls_model_weights_path,
                 description_texts_path = None,
                 vocab_size=None,  # optional for lstm
                 max_phrase_len=None):  # optional for lstm
        sen_model_name = "sentence-transformers/all-mpnet-base-v2"  # "distiluse-base-multilingual-cased-v1" - has problems with finutuning
        cls_model_name = "bert-base-uncased"
        cross_enc_model_name = 'cross-encoder/stsb-distilroberta-base'
        self.sentence_transformer = HGModelFactory.get_sentence_transformer(sen_model_name)
        self.cross_encoder = HGModelFactory.get_cross_encoder(cross_enc_model_name)

        self.tokenizer = HGModelFactory.get_hugging_face_auto_tokenizer(cls_model_name)

        #  model which is trained with ne ok calm down exercises
        self.ok_cls_model = HGModelFactory.get_hugging_face_bi_auto_model(m_name=cls_model_name,
                                                                          weights_path=ok_cls_model_weights_path)
        #  model which is trained with filtered ne ok calm down exercises
        self.neok_cls_model = HGModelFactory.get_hugging_face_bi_auto_model(m_name=cls_model_name,
                                                                            weights_path=neok_cls_model_weights_path)

        if description_texts_path is not None:
            self.desc_dict = self.create_desc_dict(description_texts_path)

    def run_inference(self,
                      text):

        preproc_text = clean_text(text)
        # preproc_text = list(map(lambda x: clean_text(x), texts))

        tokenized_ex = self.tokenizer.encode(preproc_text,
                                            truncation=True,
                                            padding="max_length",
                                            return_tensors="tf")

        ok_preds = self.ok_cls_model(tokenized_ex)
        neok_preds = self.neok_cls_model(tokenized_ex)

        embs = self.sentence_transformer.encode(text)

        desc_sim_dict = self.get_desc_category_sim(text)

        return Result(text,
                      embs,
                      ok_preds.logits[0][0].numpy(),
                      neok_preds.logits[0][0].numpy(),
                      desc_sim_dict)

    def get_desc_category_sim(self, text):
        result_dict = {}
        for key, desc_item in self.desc_dict.items():
            preds = self.cross_encoder.predict([text, desc_item["Intro message"]])
            result_dict[key] = preds

        return result_dict

    def create_desc_dict(self, desc_path):
        info_keys = ["Description", "URL", "Intro message", "Follow-up messages"]
        desc_dict = {}
        for f_path in glob.glob(os.path.join(desc_path, "*")):
            ex_type = f_path.split("/")[-1]
            desc_dict[ex_type] = {}
            with open(f_path) as file:
                s = file.read().replace('\n', '')

                for index, info_k in enumerate(info_keys):
                    end_k = info_keys[index + 1] if index < len(info_keys) - 1 else ""
                    cropped_s = s[s.find(info_k) + len(info_k):s.rfind(end_k)]
                    cropped_s = cropped_s.replace(": ", "")  # remove extra chars

                    desc_dict[ex_type][info_k] = cropped_s

        return desc_dict
