import torch 
from torch import nn 
from torch.nn import functional as F
from transformers import AutoModel
from torchcrf import CRF



class NERModel(nn.Module):
    def __init__(self, embedding_dim, num_classes, hidden_dim = 256, prototype_embeddings_dim = 8,  dropout=0.25):
        super().__init__()
        # self.base_model = base_model
        self.dropout = dropout
        
        # Projection layer
        # self.linear = nn.Linear(base_model.embeddings.word_embeddings.embedding_dim, prototype_embeddings_dim)
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(p = dropout),
            nn.Linear(256, prototype_embeddings_dim)
        )
        
        
        
        # Prototypes for each classes
        self.prottypes = nn.Linear(num_classes, prototype_embeddings_dim)


    
    def forward(self, inputs, return_embeddings = False):
        # x = self.base_model(input_ids = input_ids, attention_mask = attention_mask)
        x = F.dropout(inputs, p = self.dropout)
        x = self.linear(x)
        
        embeddings = x
        
        x = x @ self.prottypes.weight
        
        if return_embeddings:
            return x, embeddings
        return x
class EntityDetectionModel(NERModel):
    def __init__(self, embedding_dim, num_classes, hidden_dim = 256, prototype_embeddings_dim = 8,  dropout=0.25):
        super().__init__(embedding_dim=embedding_dim, num_classes=num_classes, hidden_dim=hidden_dim, prototype_embeddings_dim=prototype_embeddings_dim, dropout=dropout)
        

class NegExModel(NERModel):
    def __init__(self, embedding_dim, num_classes, hidden_dim=256, prototype_embeddings_dim = 8,  dropout=0.25):
        super().__init__(embedding_dim=embedding_dim, num_classes=num_classes, hidden_dim=hidden_dim, prototype_embeddings_dim=prototype_embeddings_dim, dropout=dropout)
        

class AssertionModel(nn.Module):
    def __init__(self, encoders, hidden_dim = 256, prototype_embeddings_dim = 8, dropout = 0.25):
        super().__init__()
        self.base_model =  AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.ner_model = NERModel(self.base_model.embeddings.word_embeddings.embedding_dim, encoders["ner"].num_tags, hidden_dim=hidden_dim, prototype_embeddings_dim=prototype_embeddings_dim, dropout=dropout)
        self.entity_detection_model = EntityDetectionModel(self.base_model.embeddings.word_embeddings.embedding_dim, encoders["detection"].num_tags, hidden_dim=hidden_dim, prototype_embeddings_dim=prototype_embeddings_dim, dropout=dropout)
        self.negex_model = NegExModel(self.base_model.embeddings.word_embeddings.embedding_dim, encoders["negex"].num_tags, hidden_dim=hidden_dim, prototype_embeddings_dim=prototype_embeddings_dim, dropout=dropout)
        self.linear =  nn.Sequential(
            nn.Linear(prototype_embeddings_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p = dropout),
            nn.Linear(256, encoders["assertion"].num_tags)
            
        )
    def forward(self, input_ids, attention_mask, return_all = False):
        
        base_outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        ner_outputs, entity_vectors = self.ner_model(base_outputs.last_hidden_state, return_embeddings = True)
        detection_outputs, detection_vectors = self.entity_detection_model(base_outputs.last_hidden_state, return_embeddings = True)
        negation_outputs, negation_vectors = self.negex_model(base_outputs.last_hidden_state, return_embeddings = True)
        
        x = torch.cat([entity_vectors, detection_vectors, negation_vectors], dim=-1)
        
        x = self.linear(x)
        if return_all:
            return x, (ner_outputs, detection_outputs, negation_outputs)
        return x

class CRFJointModel(nn.Module):
    def __init__(self, encoders,  hidden_dim = 256, prototype_embeddings_dim = 8, dropout = 0.25):
        super().__init__()
        self.base_model =  AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.ner_model = NERModel(self.base_model.embeddings.word_embeddings.embedding_dim, encoders["ner"].num_tags, hidden_dim= hidden_dim, prototype_embeddings_dim=prototype_embeddings_dim, dropout=dropout)
        self.entity_detection_model = EntityDetectionModel(self.base_model.embeddings.word_embeddings.embedding_dim, encoders["detection"].num_tags, hidden_dim, prototype_embeddings_dim=prototype_embeddings_dim, dropout=dropout)
        self.negex_model = NegExModel(self.base_model.embeddings.word_embeddings.embedding_dim, encoders["negex"].num_tags, hidden_dim, prototype_embeddings_dim=prototype_embeddings_dim, dropout=dropout)
        self.assertion_model = nn.Sequential(
            nn.Linear(prototype_embeddings_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p = dropout),
            nn.Linear(256, encoders["assertion"].num_tags)
            
        )
        
        self.ner_crf = CRF(encoders["ner"].num_tags, batch_first = True)
        self.detection_crf = CRF(encoders["detection"].num_tags, batch_first = True)
    
        self.assertion_cce_loss = nn.CrossEntropyLoss()
        self.negex_cce_loss = nn.CrossEntropyLoss()
        
        
        
        
    def forward(self, input_ids, attention_mask, attention_masks):

        base_outputs = self.base_model(input_ids, attention_mask)
        
        ner_outputs, ner_embeddings = self.ner_model(base_outputs.last_hidden_state, return_embeddings=True)
        detection_outputs, detection_embeddings = self.entity_detection_model(base_outputs.last_hidden_state, return_embeddings=True)
        negex_outputs, negex_embeddings = self.negex_model(base_outputs.last_hidden_state, return_embeddings=True)
        
        embeddings = torch.cat([ner_embeddings, detection_embeddings, negex_embeddings], dim=-1)
        
        assertion_outputs = self.assertion_model(embeddings)
        
        ner_tags = self.ner_crf.decode(ner_outputs[:, 1:], mask = attention_masks["ner"][:, 1:])
        detection_tags = self.detection_crf.decode(detection_outputs[:, 1:], mask = attention_masks["detection"][:, 1:])
        
        
        
        return {
            "ner":ner_tags,
            "detection":detection_tags,
            "negex":negex_outputs,
            "assertion":assertion_outputs
        }
        
    def nll_loss(self, input_ids, attention_mask, **targets):
       
        
        base_outputs = self.base_model(input_ids, attention_mask)
        
        ner_outputs, ner_embeddings = self.ner_model(base_outputs.last_hidden_state, return_embeddings=True)
        detection_outputs, detection_embeddings = self.entity_detection_model(base_outputs.last_hidden_state, return_embeddings=True)
        negex_outputs, negex_embeddings = self.negex_model(base_outputs.last_hidden_state, return_embeddings=True)
        
        embeddings = torch.cat([ner_embeddings, detection_embeddings, negex_embeddings], dim=-1)
        
        assertion_outputs = self.assertion_model(embeddings)
        
        loss1 = self.ner_crf(ner_outputs[:, 1:], targets["ner_labels"][:, 1:], mask = targets["attention_masks"]["ner"][:, 1:])

        loss2 = self.detection_crf(detection_outputs[:, 1:], targets["detection_labels"][:, 1:], mask = targets["attention_masks"]["detection"][:, 1:])
        loss3 = self.negex_cce_loss(negex_outputs.permute(0, 2, 1), targets["negex_labels"])
        loss4 = self.assertion_cce_loss(assertion_outputs.permute(0, 2, 1), targets["assertion_labels"])
        
        return -loss1 - loss2 + loss3 * input_ids.shape[0] + loss4 * input_ids.shape[0]
        
class PHIModel(nn.Module):
    def __init__(self, encoder, hidden_dim = 256, prototype_embeddings_dim = 8, dropout = 0.25):
        super().__init__()
        self.base_model =  AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.ner_model = NERModel(self.base_model.embeddings.word_embeddings.embedding_dim, encoder.num_tags, hidden_dim=hidden_dim, prototype_embeddings_dim=prototype_embeddings_dim, dropout=dropout)
    def forward(self, **inputs):
        outputs = self.base_model(**inputs)
        return self.ner_model(outputs.last_hidden_state)