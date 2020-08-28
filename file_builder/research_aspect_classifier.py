import torch
import numpy as np
import torch

from transformers import TFBertForSequenceClassification, BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification, PretrainedConfig, BertConfig


class ResearchAspectClassifier:
    def __init__(self, aspects):
        self.aspects = aspects
        
        self.bert_config = BertConfig(vocab_size=31090, num_labels=5)
        
        
        self.model = AutoModelForSequenceClassification.from_pretrained('allenai/scibert_scivocab_uncased', config=self.bert_config).to(torch.device('cpu'))
        
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

        #Load model from path
        self.model.load_state_dict(torch.load("file_builder/scibert_best_model.pt", map_location=torch.device('cpu')))
        
        self.model.eval()
        print('== Done Loading ResearchAspectClassifier ==')
        
    def classify_abstract(self, sentences):
        classified_abstract = {}
        for research_aspect in self.aspects:
            classified_abstract[research_aspect] = []

        for z, sentence in enumerate(sentences):
            classified_abstract[self.aspects[self._return_aspect_sentence(sentence)]].append((z, sentence))
            
        return classified_abstract
        
        
    def _return_aspect_sentence(self, sentence):
        input_ids = torch.tensor(self.tokenizer.encode(sentence, add_special_tokens=False)).unsqueeze(0)  # Batch size 1
        outputs = self.model(input_ids)[0]

        values = torch.max(outputs, 1)[1]
        cpu_pred = outputs.cpu()
        result = cpu_pred.data.numpy()
        array_res = np.reshape(result, (5,1))
        for i in range(0, len(array_res)):
            if array_res[i] == max(array_res):
                return i

        return -1