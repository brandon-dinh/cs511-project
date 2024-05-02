
import os
import tasti
import jsonlines
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict

from datasets import Dataset, load_dataset
from PIL import Image
import requests
from transformers import AutoProcessor

from torch.utils.data import DataLoader

from transformers import BlipProcessor, BlipPreTrainedModel, BlipConfig, BlipVisionModel, BlipTextModel
from transformers.models.blip.modeling_blip import BlipImageTextMatchingModelOutput, BlipTextVisionModelOutput

import torch
import torch.nn as nn
from torch.nn.functional import normalize

from typing import Optional, Tuple, Union

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BlipForImageTextRetrieval(BlipPreTrainedModel):
    config_class = BlipConfig

    def __init__(self, config: BlipConfig):
        super().__init__(config)

        self.vision_model = BlipVisionModel(config.vision_config)

        self.text_encoder = BlipTextModel(config.text_config, add_pooling_layer=False)

        # vision projection layer
        self.vision_proj = nn.Linear(config.vision_config.hidden_size, config.image_text_hidden_size)

        # text projection layer
        self.text_proj = nn.Linear(config.text_config.hidden_size, config.image_text_hidden_size)

        # image text matching head
        self.itm_head = nn.Linear(config.text_config.hidden_size, 2)

        self.decoder_pad_token_id = (
            config.text_config.pad_token_id
            if not hasattr(config, "decoder_pad_token_id")
            else config.decoder_pad_token_id
        )
        self.decoder_start_token_id = (
            config.text_config.bos_token_id
            if not hasattr(config, "decoder_start_token_id")
            else config.decoder_start_token_id
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        use_itm_head: Optional[bool] = True,
        attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BlipTextVisionModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, BlipForImageTextRetrieval

        >>> model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-base-coco")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "an image of a cat"

        >>> inputs = processor(images=image, text=text, return_tensors="pt")
        >>> outputs = model(**inputs)
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[0]
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long)

        if use_itm_head:
            question_embeds = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=return_dict,
            )
            question_embeds = question_embeds[0] if not return_dict else question_embeds.last_hidden_state

            output = self.itm_head(question_embeds[:, 0, :])
        else:
            question_embeds = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=return_dict,
            )
            question_embeds = question_embeds[0] if not return_dict else question_embeds.last_hidden_state

            image_feat = normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
            text_feat = normalize(self.text_proj(question_embeds[:, 0, :]), dim=-1)

            output = image_feat @ text_feat.t()

        if not return_dict:
            outputs = (output, vision_outputs[0]) + vision_outputs[2:] + (question_embeds,)
            return tuple(output for output in outputs if output is not None)

        return BlipImageTextMatchingModelOutput(
            itm_score=output,
            last_hidden_state=vision_outputs.last_hidden_state,
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
            question_embeds=question_embeds,
        )

class BlipVisionWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        image_embeds = self.model.vision_model(x['pixel_values'].to(device))[0]
        image_feat = normalize(self.model.vision_proj(image_embeds[:, 0, :]), dim=-1)
        return image_feat
    
class BlipTextImageWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        y = self.model(pixel_values=x['pixel_values'].unsqueeze(0).to(device), input_ids=x['input_ids'].unsqueeze(0).to(device), attention_mask=x['attention_mask'].unsqueeze(0).to(device))[0]
        itm_score = torch.nn.functional.softmax(y, dim=1)[:, 1]
        print("image_id, itm_score", x['image_id'], itm_score)
        return itm_score.cpu().detach().numpy()

class QASIIndex(tasti.Index):
    def __init__(self, config, dataset):
        self.dataset = dataset

        self.model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
        self.model.eval()
        self.model.to(device)

        self.vision_wrapper = BlipVisionWrapper(self.model)
        self.embedding_wrapper = BlipTextImageWrapper(self.model)
        super().__init__(config)

    def get_target_dnn(self):
        return self.embedding_wrapper
    
    def get_pretrained_embedding_dnn(self):
        return self.vision_wrapper
        
    def get_embedding_dnn(self):
        model = torch.nn.Identity()
        return model
    
    def get_target_dnn_dataset(self, train_or_test):
        return dataset
    
    def get_embedding_dnn_dataset(self, train_or_test):
        return dataset
    
class QASISUPGPrecisionQuery(tasti.SUPGPrecisionQuery):
    def __init__(self, index):
        super().__init__(index)
        self.count = 0

    def score(self, target_dnn_output):
        # print("score", target_dnn_output)
        # return target_dnn_output
        return 1.0 if target_dnn_output > 0.1 else 0.0

# class QASISUPGRecallQuery(tasti.SUPGRecallQuery):
#     def __init__(self, index):
#         super().__init__(index)
#         self.count = 0

#     def score(self, target_dnn_output):
#         # print("score", target_dnn_output)
#         return 1.0 if target_dnn_output > 0.3 else 0.0

class QASIConfig(tasti.IndexConfig):
    def __init__(self):
        super().__init__()
        self.do_mining = False
        self.do_training = False
        self.do_infer = False
        self.do_bucketting = True
        self.nb_train = 500000
        self.nb_buckets = 500
        self.batch_size = 64

def init_dataset(query_text='vase on wooden table'):
    processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-base-coco")

    def transform(example):
        inputs = processor(images=example['image'], text=[query_text] * len(example['image']), return_tensors="pt")
        inputs['image_id'] = example['image_id']
        return inputs

    dataset = load_dataset("visual_genome", "attributes_v1.2.0", split='train').shuffle().select(range(5000))
    print(dataset)
    dataset.set_transform(transform)
    return dataset

if __name__ == '__main__':
    config = QASIConfig()
    dataset = init_dataset("scenes where individuals are engaged in outdoor sports, such as playing basketball outside, skateboarding down city sidewalks, or practicing yoga in serene park settings")
    index = QASIIndex(config, dataset)
    index.init()
    query = QASISUPGPrecisionQuery(index)
    result = query.execute_metrics(budget=500)
    print(result)
