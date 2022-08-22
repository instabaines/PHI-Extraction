from torch.utils.data import Dataset, DataLoader
import tqdm
import pandas as pd
import torch
import IPython
import os
def prepare_sentence(tokenizer, record, encoders, max_length):
    words = record["words"]
    
    ner_tags = record["NER_tags"]
    assertion_tags = record["Assertion_tags"]
    negation_tags = record["Negex_tags"]
    

    output_tokens = ["[CLS]"]
    output_ner_labels = [encoders["ner"].pad_label_id]
    output_detection_labels = [encoders["detection"].pad_label_id]
    output_negex_labels = [encoders["negex"].pad_label_id]
    output_assertion_labels = [encoders["assertion"].pad_label_id]
        
    for i in range(len(words)):
        
        word = words[i]
        ner_tag = ner_tags[i]
        assertion_tag = assertion_tags[i]
        
        tokens = tokenizer.tokenize(word)
        first = True
        
        
        for token in tokens:
            if not first:
                ner_tag = ner_tag.replace("B-", "I-")
                if type(assertion_tag) == str:
                    assertion_tag = assertion_tag.replace("B-", "I-")
               
                if ner_tag == "O":
                    detection_tag = "O"
                else:
                    detection_tag = "I-ENT"
                
                if negation_tags[i]!="" and ner_tag!="O":
                    negex_tag = "I-{}".format(negation_tags[i].upper())
                else:
                    negex_tag = None
            else:
                first = False
                if ner_tag=="O":
                    detection_tag = "O"
                    negex_tag = None
                elif ner_tag.startswith("B-"):
                    detection_tag = "B-ENT"
                    if negation_tags[i]!="":
                        negex_tag = "B-{}".format(negation_tags[i].upper())
                    else:
                        negex_tag = None
                else:
                    detection_tag = "I-ENT"
                    if negation_tags[i]!="":
                        negex_tag = "I-{}".format(negation_tags[i].upper())
                    else:
                        negex_tag = None
                    
            
            
            output_tokens.append(token)
            output_ner_labels.append(encoders["ner"].encode(ner_tag))
            if type(assertion_tag) == str and assertion_tag!="":
                output_assertion_labels.append(encoders["assertion"].encode(assertion_tag))
            else:
                output_assertion_labels.append(encoders["assertion"].pad_label_id)
            
            output_detection_labels.append(encoders["detection"].encode(detection_tag))
            if type(negex_tag) == str and negex_tag !="":
                output_negex_labels.append(encoders["negex"].encode(negex_tag))
            else:
                output_negex_labels.append(encoders["negex"].pad_label_id)
            
            
    output_tokens.append("[SEP]")
    output_assertion_labels.append(encoders["assertion"].pad_label_id) # PAD sep, we don't want to compute error for sep
    output_detection_labels.append(encoders["detection"].pad_label_id) 
    output_negex_labels.append(encoders["negex"].pad_label_id) 
    output_ner_labels.append(encoders["ner"].pad_label_id) 
    
    input_ids = tokenizer.convert_tokens_to_ids(output_tokens)
    mask = [1]  * len(input_ids)
    
    while len(input_ids) < max_length:
        input_ids.extend(tokenizer.convert_tokens_to_ids(["[PAD]"]))
        mask.append(0)
        output_assertion_labels.append(encoders["assertion"].pad_label_id) # PAD sep, we don't want to compute error for sep
        output_detection_labels.append(encoders["detection"].pad_label_id) 
        output_negex_labels.append(encoders["negex"].pad_label_id) 
        output_ner_labels.append(encoders["ner"].pad_label_id) 
    
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        mask = mask[:max_length]
        output_assertion_labels = output_assertion_labels[:max_length]
        output_detection_labels = output_detection_labels[:max_length]
        output_negex_labels = output_negex_labels[:max_length]
        output_ner_labels = output_ner_labels[:max_length]
        
        
    ner_mask = torch.tensor(output_ner_labels)!=encoders["ner"].pad_label_id
    detection_mask = torch.tensor(output_detection_labels)!=encoders["detection"].pad_label_id


    
    return {
        "input_ids":torch.tensor(input_ids),
        "attention_mask":torch.tensor(mask),
        "attention_masks":{
            "ner": ner_mask,
            "detection": detection_mask,            
            },
        "assertion_labels":torch.tensor(output_assertion_labels),
        "detection_labels":torch.tensor(output_detection_labels),
        "negex_labels":torch.tensor(output_negex_labels),
        "ner_labels":torch.tensor(output_ner_labels)
    }        




class I2b2NERDatset(Dataset):
    def __init__(self, path, encoders, tokenizer, max_length = 128):
      
        self.encoders = encoders
        self.tokenizer = tokenizer
        self.max_length = max_length

        df = pd.read_csv(path, keep_default_na=False)
        
        
   
        
        lines = {}
        print("Loading dataset")
        for index, row in tqdm.auto.tqdm(df.iterrows(), total=len(df)):
            key = "{}-{}".format(row["doc_id"], row["line_number"])
            lines.setdefault(key, {"words":[], "NER_tags":[], "Assertion_tags":[], "Negex_tags":[]})
            lines[key]["words"].append(row["word"])
            lines[key]["NER_tags"].append(row["NER_tag"])
            lines[key]["Assertion_tags"].append(row["Assertion_tag"])
            lines[key]["Negex_tags"].append(row["negex_assertion"])
        
        self.records = []
        for doc_id in lines:
            
            self.records.append(lines[doc_id])
        
        
    def __len__(self):
        return len(self.records)
    def __getitem__(self, index):
        
        record = self.records[index]
        
        output = prepare_sentence(self.tokenizer, record, self.encoders,  self.max_length)
        
        return output
    
