from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
from typing import Iterator, List, Dict
import torch
import torch.optim as optim
import numpy as np
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField, LabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor

torch.manual_seed(1)


# def findFiles(path): return glob.glob(path)

# print(findFiles('data/names/*.txt'))
#@DatasetReader.register("datasetreader")
class PosDatasetReader(DatasetReader):
    """
    DatasetReader for PoS tagging data, one sentence per line, like

        The###DET dog###NN ate###V the###DET apple###NN
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
   
    def text_to_instance(self, tokens: List[Token], tags: List[str] = None) -> Instance:
        tags = tags*len(list(tokens))
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}

        if tags:
            label_field = LabelField(tags)
            fields["labels"] = label_field

        return Instance(fields)
#      def text_to_instance(self, tokens: List[Token], id: str,
#                          labels: np.ndarray) -> Instance:
#         sentence_field = TextField(tokens, self.token_indexers)
#         fields = {"tokens": sentence_field}
        
#         id_field = MetadataField(id)
#         fields["id"] = id_field
        
#         label_field = ArrayField(array=labels)
#         fields["label"] = label_field

#         return Instance(fields)
    

    def findFiles(self,path): return glob.glob(path)
    
    
    all_letters = string.ascii_letters + " .,;'"
    n_letters = len(all_letters)
    
    
    def unicodeToAscii(self,s):
     return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in self.all_letters
     )


    category_lines = {}
    all_categories = []

    # Read a file and split into lines
    def readLines(self,filename):
      lines = open(filename, encoding='utf-8').read().strip().split('\n')
      return [self.unicodeToAscii(line) for line in lines]
    
    def _read(self, file_path: str)-> Iterator[Instance]:
      for filename in self.findFiles(file_path):
#             print(filename)

            category = os.path.splitext(os.path.basename(filename))[0]
#             print (category)
            self.all_categories.append(category)
            lines = self.readLines(filename)
#             print (lines)
            #for line in lines:
            yield self.text_to_instance([Token(line) for line in lines], category)
            self.category_lines[category] = lines
           # print(self.category_lines[category][:5])
#             print(filename)

#      n_categories = len(self.all_categories)
     
