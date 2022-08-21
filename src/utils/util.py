import numpy as np 
import torch 
from datasets import load_dataset, load_metric

seqeval_metrics =  load_metric("seqeval")



def get_word_tags(tokenizer, tokens, preds_probs, labels = None):
    output_words = []
    output_pred_tags = []
    if labels is not None:
        output_label_tags = []
    
    current_probs = None
    if labels is not None:
        current_label = None
    current_word  = None
    
    
    index = 0
    while index < len(tokens):
        word = tokenizer.decode(int(tokens[index]))
        if word in ["[PAD]", "[CLS]", "[SEP]"]:
            index += 1
            continue
        if current_word is None:
            assert not (word.startswith("##"))
            current_word = word
            current_probs = [preds_probs[index]]
            if labels is not None:
                current_label = labels[index]
        else:
            if word.startswith("##"):
                current_word += word[2:]
                current_probs.append(preds_probs[index])
            else:
                mean_probs = np.mean(current_probs, axis=0)
                output_words.append(current_word)
                output_pred_tags.append(np.argmax(mean_probs))
                if labels is not None:
                    output_label_tags.append(current_label)
                
                current_word = word
                current_probs = [preds_probs[index]]
                if labels is not None:
                    current_label = labels[index]
        index += 1
    if current_word != "" and current_word is not None:
        mean_probs = np.mean(current_probs, axis=0)
        output_words.append(current_word)
        output_pred_tags.append(np.argmax(mean_probs))
        if labels is not None:
            output_label_tags.append(current_label)

       
    if labels is not None:
        return output_words, output_pred_tags, output_label_tags
    else:
        return output_words, output_pred_tags
        
                

def compute_metrics(p, dataset, tag_inverse_mapping, tokenizer):
    predictions, labels = p

    tokens = [d["input_ids"] for d in dataset]

    new_predictions, new_labels = [], []
    for tkns, preds, lbls in zip(tokens, predictions, labels):
        _, preds, lbls = get_word_tags(tokenizer, tkns, preds, lbls)
        
        new_predictions.append(preds)
        new_labels.append(lbls)

    predictions = new_predictions
    labels = new_labels

    
    true_predictions = [
        [tag_inverse_mapping[p] for (p, l) in zip(prediction, label) if l!=-100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [tag_inverse_mapping[l]for (p, l) in zip(prediction, label) if l!=-100]
        for prediction, label in zip(predictions, labels)
    ]

    
   
    result =  seqeval_metrics.compute(predictions=true_predictions, references=true_labels, scheme="IOB2", zero_division=0.0)
    
    return {
        "precision": result["overall_precision"],
        "recall": result["overall_recall"],
        "f1": result["overall_f1"],
        "accuracy": result["overall_accuracy"],
    }