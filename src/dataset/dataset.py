from torch.utils.data import Dataset, DataLoader
import tqdm
import pandas as pd
import torch
import IPython
import os
import re 



def tokenize_and_preserve_labels(tokenizer,sentence, text_labels,tag2idx):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):
        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)
        out_label=[]
        first = True
        for token in tokenized_word:
          if not first:
            label = label.replace("B-", "I-")
            out_label.append(tag2idx[label])
          else:
            first=False
            out_label.append(tag2idx[label])
        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend(out_label)

    return tokenized_sentence, labels
def prepare_sentence(tokenizer, sentence,text_labels, tag2idx, max_length):

    output_tokens = ["[CLS]"]
    output_labels = [tag2idx['PAD']]
    tokenized_sentence,labels = tokenize_and_preserve_labels(tokenizer,sentence,text_labels,tag2idx)
    output_tokens.extend(tokenized_sentence)
    output_labels.extend(labels)
    output_tokens.append("[SEP]")
    output_labels.append(tag2idx['PAD']) 
    
    input_ids = tokenizer.convert_tokens_to_ids(output_tokens)
    mask = [1]  * len(input_ids)
    
    while len(input_ids) < max_length:
        input_ids.extend(tokenizer.convert_tokens_to_ids(["[PAD]"]))
        output_labels.append(tag2idx['PAD'])
        mask.append(0)
    
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        mask = mask[:max_length]
        output_labels = output_labels[:max_length]
       
        
     
    
    return {
        "input_ids":torch.tensor(input_ids),
        "attention_mask":torch.tensor(mask),
       
        "labels":torch.tensor(output_labels)
    }        



class PHIDataset(Dataset):
    def __init__(self, tag2idx, sentences, labels, tokenizer, max_length = 128):
        super().__init__()
        self.sentences=sentences
        self.labels = labels
        self.tag2idx = tag2idx
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, index):
        sentence = self.sentences[index]
        text_labels= self.labels[index]
        
        return prepare_sentence(self.tokenizer, sentence,text_labels, self.tag2idx, self.max_length)
