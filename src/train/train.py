from src.models import PHIModel
from src.dataset import PHIDataset
import argparse
import pandas as pd
import os
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
import tqdm
from datasets import load_dataset, load_metric
from seqeval.scheme import IOB2
from collections import Counter
from sklearn.metrics import classification_report
import itertools
import itertools
from src.utils import TagEncoder
import IPython

seqeval_metric = load_metric("seqeval")
pad_label_id = -100

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=str, help="Dataset directory")
    parser.add_argument("--log-dir", default="./logs", type=str, help="Dataset directory")
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument('--freeze', dest='freeze', action='store_true')
    parser.add_argument('--epochs',  type=int, default=10, help="Number of epochs to train the model")
    parser.add_argument('--batch-size',  type=int, default=32, help="Batch size for the training")
    parser.add_argument('--lr',  type=float, default=1e-5, help="Learning rate for the training")
    
    
    args = parser.parse_args()
    
    
    return args
def get_tag_encoder(dataset_dir, tags2consider):
    df  = pd.read_csv(os.path.join(dataset_dir, "train.csv"))
    df.tag = df.tag.apply(lambda x: x.split("_")[0])
    df.tag = df.tag.apply(lambda x: x if x in tags2consider else "O")
    tags = set()
    for tag in df.tag.unique():
        if tag!= "O":
            tags.add("B-"+tag)
            tags.add("I-"+tag)
        else:
            tags.add(tag)
    tags = list(tags)
    tags.sort()
    return TagEncoder(tags, pad_label_id = -100)
    

def train_model(args, model, loader, optimizer, criterion, device):
    model.train()
    losses = 0.0
    total = 0.0

    for batch in tqdm.auto.tqdm(loader, total = len(loader)):
        for key in batch.keys():
            batch[key] = batch[key].to(device)
        outputs =  model(input_ids = batch["input_ids"], attention_mask  = batch["attention_mask"])
       
        
        
        
        loss = criterion(outputs.permute(0, 2, 1), batch["labels"])
        
                
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        losses += loss.item() * batch["labels"].shape[0]
        
        total +=   batch["labels"].shape[0]
    return losses/total
    

def evaluate_model(args, model, loader, criterion, device):
    model.eval()
    losses = 0.0
    total = 0.0
    
    actual_labels = []
    predicted_labels = []
    
        
    with torch.no_grad():

        for batch in tqdm.auto.tqdm(loader, total = len(loader)):
            for key in batch.keys():
                batch[key] = batch[key].to(device)
                
            outputs =  model(input_ids = batch["input_ids"], attention_mask  = batch["attention_mask"])
        
        
        
            loss = criterion(outputs.permute(0, 2, 1), batch["labels"])
            predictions = outputs.argmax(dim=-1)
            actual_labels.extend([[loader.dataset.encoder.decode(l.item()) for l in labels if l!=pad_label_id] for labels in batch["labels"]])
            predicted_labels.extend([[loader.dataset.encoder.decode(p.item()) for p, l in zip(preds, labels) if l!=pad_label_id] for (preds, labels) in zip(predictions, batch["labels"])])
            
            
            
            losses += loss.item() * batch["input_ids"].shape[0]
            
            total += batch["input_ids"].shape[0]
            
            
    result =  seqeval_metric.compute(predictions=predicted_labels, references=actual_labels, zero_division=0.0)
    
    
   

   
        
    return losses/total, result


def training_loop(args, model, trainloader, valid_loader, optimizer, criterion, num_epochs):
    print("Started training loop")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    for epoch in range(num_epochs):
        train_loss = train_model(args, model, trainloader, optimizer, criterion, device)
     
        valid_loss, valid_metrics = evaluate_model(args, model, valid_loader, criterion, device)
        print("Epoch: {}/{} train-loss: {:.4f} valid-loss: {:.4f}".format(epoch + 1, num_epochs, train_loss, valid_loss))
        print("Ner: Precision: {:.4f} Recall: {:.4f} F1-Score: {:.4f} Accuracy: {:.4f}".format(
            valid_metrics["overall_precision"], valid_metrics["overall_recall"],
            valid_metrics["overall_f1"], valid_metrics["overall_accuracy"]))
       
        
    torch.save({"model":model}, os.path.join(args.log_dir, "phi-model.pt"))
def main(args):
    
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    
    tags2consider = {'AGE', 'CONTACT', 'DATE', 'ID', 'LOCATION', 'NAME', 'O', 'PHI', 'PROFESSION'}
    encoder = get_tag_encoder(args.dataset_dir, tags2consider)


    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    
    train_dataset = PHIDataset(os.path.join(args.dataset_dir, "train.csv"),encoder, tokenizer,  max_length = 512, tags2consider = tags2consider)
    valid_dataset = PHIDataset(os.path.join(args.dataset_dir, "valid.csv"),encoder, tokenizer,  max_length = 512, tags2consider = tags2consider)
    
    if args.resume:
        print("Resuming from previously saved model")
        state = torch.load(os.path.join(args.log_dir, "phi-model.pt"))
        model = state["model"]
    else:
        model = PHIModel(encoder)
    
    if args.freeze:
        print("Freezing base model parameters")
        for param in model.base_model.parameters():
            param.requires_grad = False

        
    
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr)
        
    
    criterion = torch.nn.CrossEntropyLoss()
    
    
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size = args.batch_size, shuffle=False, drop_last = False)
    
    training_loop(args, model, train_loader, valid_loader, optimizer, criterion, args.epochs)
    
    

if __name__ == '__main__':
    args = get_args()
    main(args)