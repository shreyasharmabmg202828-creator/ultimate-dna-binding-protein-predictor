import os
import sys
import torch
import joblib
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Import your engineered feature functions from your neighboring script
from features import encode_bert, get_charge_features, get_structural_features

def main():
    print("=" * 60)
    print(" Initializing Ultimate v3 DNA-Binding Predictor Runtime")
    print("=" * 60)
    
    # 1. Device selection (GPU Acceleration if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Running inference engine on device: {device}")
    
    # 2. Paths to your saved production binaries
    model_dir = "dna_binding_final"
    model_path = os.path.join(model_dir, "final_v3_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler_u.pkl")
    
    # Verify assets exist before trying to load them
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        print(f"\n[Error] Model assets not found in folder '{model_dir}/'.")
        print("Please ensure your trained .pkl files are uploaded correctly.")
        sys.exit(1)
        
    # 3. Load ML model parameters and normalization scales
    print("[*] Materializing trained weights and scaling parameters...")
    model_v3 = joblib.load(model_path)
    scaler_u = joblib.load(scaler_path)
    
    # 4. Load the ProtBERT language framework
    print("[*] Connecting to Rostlab/prot_bert transformer core...")
    tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    bert_model = AutoModel.from_pretrained("Rostlab/prot_bert").to(device)
    
    print("\n✅ System fully online and ready for processing.")
    print("Type 'exit' to shut down the runtime loop.\n")
    
    while True:
        # Capture raw sequence string from console terminal
        user_input = input("Enter Protein Sequence: ").strip().upper()
        if user_input == 'EXIT':
            print("Shutting down engine runtime. Goodbye!")
            break
            
        # Basic validation checklist
        standard_aa = set("ACDEFGHIKLMNPQRSTVWY")
        if not all(aa in standard_aa for aa in user_input) or len(user_input) == 0:
            print("❌ Input validation error: Standard IUPAC amino acids only.\n")
            continue
            
        print("  Analyzing peptide strings across biological tiers...")
        
        try:
            # Tier A: Compute high-dimensional semantic context vector
            b = encode_bert(user_input, tokenizer, bert_model, device).reshape(1, -1)
            
            # Tier B & C: Compute physical traits and structural propensities 
            p = np.array(get_charge_features(user_input)).reshape(1, -1)
            s = np.array(get_structural_features(user_input)).reshape(1, -1)
            
            # Unify all tiers into the full 1030D vector and scale
            feat_vector = np.hstack((b, p, s))
            feat_scaled = scaler_u.transform(feat_vector)
            
            # Compute classification probability using the trained Support Vector machine
            prob = model_v3.predict_proba(feat_scaled)[0][1]
            
            # --- CRITICAL BIOLOGY-GUIDED CONSTRAINTS (Bio-Refinement) ---
            
            # Checklist item 1: The Histone Fix (Dynamic Threshold Adaptation)
            basic_content = (user_input.count('R') + user_input.count('K')) / len(user_input)
            current_threshold = 0.50 if basic_content > 0.18 else 0.60
            
            # Checklist item 2: The Collagen Guard (Turn propensity filtering)
            if s[0][2] > 0.35:
                label = "NEGATIVE (Structural Mimic Detected)"
                confidence = prob * 0.5  # Penalize output score confidence
            else:
                label = "POSITIVE (DNA-BINDING)" if prob > current_threshold else "NEGATIVE (NON-BINDING)"
                confidence = prob
                
            # 5. Format and show prediction breakdown
            print("\n" + "="*50)
            print("         ULTIMATE V3 BIO-HYBRID PREDICTION        ")
            print("="*50)
            print(f"Sequence Snippet  : {user_input[:30]}...")
            print(f"Total Residues    : {len(user_input)}")
            print("-" * 50)
            
            if "POSITIVE" in label:
                print(f"🟢 Prediction     : {label}")
            elif "Mimic" in label:
                print(f"⚠️ Prediction     : {label}")
            else:
                print(f"❌ Prediction     : {label}")
                
            print(f"Confidence Level  : {confidence * 100:.2f}%")
            print("="*50 + "\n")
            
        except Exception as e:
            print(f"An anomaly occurred during feature extraction processing: {str(e)}\n")

if __name__ == "__main__":
    main()
