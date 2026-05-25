import numpy as np
import torch
import itertools

# 1. Evolutionary & Semantic Context (ProtBERT Engine)
def encode_bert(sequence, tokenizer, bert_model, device):
    """
    Transforms a raw protein sequence into a 1024-dimensional biological vector 
    using the mean of the last hidden state of ProtBERT.
    """
    # ProtBERT requires spaces between individual amino acid characters
    spaced_sequence = " ".join(list(sequence))
    inputs = tokenizer(spaced_sequence, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = bert_model(**inputs)
        
    # Pool tokens by taking the average across the sequence length dimension
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

# 2. Electrostatic & Physical Properties
def get_charge_features(sequence):
    """
    Calculates sequence physical metrics: Positive charge composition, 
    Negative charge composition, and Charge Transition Frequencies.
    """
    pos_group = set("KRH")  # Lysine, Arginine, Histidine
    neg_group = set("DE")   # Aspartic Acid, Glutamic Acid
    seq_len = len(sequence)
    
    if seq_len == 0:
        return [0.0, 0.0, 0.0]
        
    pos_count = sum(1 for aa in sequence if aa in pos_group) / seq_len
    neg_count = sum(1 for aa in sequence if aa in neg_group) / seq_len
    
    # Track how frequently charges alternate across adjacent residues
    transitions = 0
    for i in range(seq_len - 1):
        if (sequence[i] in pos_group and sequence[i+1] in neg_group) or \
           (sequence[i] in neg_group and sequence[i+1] in pos_group):
            transitions += 1
            
    trans_feat = transitions / (seq_len - 1) if seq_len > 1 else 0.0
    return [pos_count, neg_count, trans_feat]

# 3. Structural Propensities (Levitt Scales)
def get_structural_features(sequence):
    """
    Computes structural propensities based on Levitt's architectural scales:
    Helix-favoring (ALMQRK), Sheet-favoring (VIYWF), and Turn-favoring (GPNSD) residues.
    """
    h_fav = set("ALMQRK")
    s_fav = set("VIYWF")
    t_fav = set("GPNSD")
    seq_len = len(sequence)
    
    if seq_len == 0:
        return [0.0, 0.0, 0.0]
        
    h_content = sum(1 for aa in sequence if aa in h_fav) / seq_len
    s_content = sum(1 for aa in sequence if aa in s_fav) / seq_len
    t_content = sum(1 for aa in sequence if aa in t_fav) / seq_len
    
    return [h_content, s_content, t_content]
