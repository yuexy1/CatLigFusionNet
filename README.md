# CatLigFusionNet

CatLigFusionNet is a multimodal catalyst-ligand representation framework for yield prediction in aromatic C-H borylation.

## Purpose

This project combines experimental yield data, calculated IR descriptors, and molecular/global features to support catalyst-ligand machine-learning workflows.

## Input Data

- Experimental yield data
- Calculated IR descriptors
- Molecular and global features

## Main Scripts

- `1. Match_Exp_and_Calc.py`: matches experimental records with calculated molecular data.
- `2. Mv_logs.py`: copies or organizes Gaussian log files used by downstream descriptor extraction.
- `3. Extract_IR_descriptors.py`: extracts and processes IR spectral descriptors from log files.
- `4. Merge_data_for_training.py`: merges experimental data and descriptor tables into model-ready training data.
- `7.1 Gen_manual_GNN_emb_feats.py`: generates manual GNN embedding features and model-related descriptors.
- `7.Get_global_info.ipynb`: extracts molecular/global information from log files in notebook form.

## Data Folder

The `data/` folder contains CSV datasets used by the CatLigFusionNet workflow.

## Basic Workflow

1. Match experimental and calculated molecular records.
2. Organize required log files.
3. Extract IR spectral descriptors.
4. Merge descriptors and experimental data for training.
5. Generate GNN/manual embedding features and global molecular information.

## Notes

`steps.docx` is an internal training/instruction document and is intentionally not uploaded to GitHub.

Dependencies should be installed according to the user's local environment. The dependencies that can be identified from the current scripts are listed in `requirements.txt`.
