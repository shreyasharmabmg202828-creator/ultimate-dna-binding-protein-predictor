# Ultimate v3 DNA-Binding Protein Predictor

An advanced bio-hybrid machine learning pipeline that predicts whether a given protein sequence is a DNA-binding or non-binding protein. This architecture blends deep semantic features from a state-of-the-art Protein Language Model (pLM) with handcrafted physical and structural feature engineering to deliver highly robust, biology-guided predictions.

## 🚀 Key Performance Metrics
Compared to standard machine learning baselines and overfitted deep learning approaches, the **Ultimate v3 Hybrid Model** achieves research-grade performance on the PDB14189 dataset:

| Metric | Baseline (AAC) | Ultimate v3 Hybrid Model |
| :--- | :---: | :---: |
| **Accuracy** | 81.00% | **92.50%** |
| **Matthews Correlation (MCC)** | 0.5091 | **0.8499** |
| **F1-Score (Balanced)** | 0.8100 | **0.9275** |
| **ROC-AUC (Separation)** | — | **0.9752** |

## 🧠 Architecture & Feature Engineering
The pipeline scales a sequence into a **1030-dimensional unified feature vector** across three distinct biological tiers:

1. **Evolutionary & Semantic Context (1024 Features):** Extracted from the final hidden state mean pooling of **Rostlab's ProtBERT** language model, capturing structural and evolutionary constraints.
2. **Electrostatic & Physical Properties (3 Features):** Handcrafted sequence processing capturing positive/negative charge composition and charge transition frequencies across the polypeptide chain.
3. **Structural Propensities (3 Features):** Calculated using Levitt’s secondary structure scales to evaluate localized Helix-favoring, Sheet-favoring, and Turn-favoring amino acid densities.

---

## 🛡️ Biology-Guided Constraints (Bio-Refinement)
Standard deep learning architectures often fall victim to structural mimics (e.g., highly repetitive structural proteins or highly basic non-binding proteins). This model implements explicit algorithmic guardrails to prevent type-I and type-II errors:

### 1. The "Collagen Guard" (Structural Mimic Detection)
Repetitive structural motifs like Collagen or Elastin exhibit exceptionally high turn propensities that often trigger false-positive binding predictions. The architecture checks if Turn-favoring residues cross a threshold (`> 0.35`):
* **Action:** Re-classifies the target sequence as a structural mimic and penalizes the confidence score dynamically.

### 2. The Histone Fix (Dynamic Thresholding)
Highly basic proteins (like Histones) possess massive concentrations of Lysine (K) and Arginine (R) which naturally attract DNA electrostatically, leading to validation confusion.
* **Action:** Evaluates basic residue density (`> 18%`). If a protein is highly basic, the classification threshold is lowered dynamically from `0.60` to `0.50` to maintain sensitivity.

---

## 🛠️ Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/ultimate-protein-predictor.git](https://github.com/YOUR_USERNAME/ultimate-protein-predictor.git)
cd ultimate-protein-predictor

pip install -r requirements.txt

python src/predictor.py
