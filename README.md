# ggnn-seq2seq
Synthesizing Programs from Logical Constraints using Deep Neural Networks. Modeled the constraints in First Order Logic using Gated Graph Neural Network. Learned the Embedding in a Multi-Modal setup. Finally, used a GRU Decoder to generate the corresponding programs to satisfy the constraints.

## Requirements
- PyTorch + Cuda
- torchtext
- nltk
- typing
- time
- math
- json
- numpy
- matplotlib

## Data
- dataset_multimodal.json
Data Generation Pipeline was built upon [https://github.com/rrlcs/random-constraint-generation](Random Constraint Generation)

## Download Dataset
```
cd data
Download from https://drive.google.com/file/d/1sfh4jp2YqH-A9_Iy8zF4QHzNCVjzfmWx/view?usp=sharing
```

## Train the model
```
python3 train.py
```

## Evaluate the model
```
python3 evaluate.py
```
