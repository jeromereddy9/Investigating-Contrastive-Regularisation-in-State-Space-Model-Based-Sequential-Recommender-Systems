# Investigating Contrastive Regularization in State Space Model Based Sequential Recommender Systems

## Abstract

Contrastive learning (CL) has been shown to improve attention-based sequential recommenders, and has recently been applied to State Space Model (SSM)-based recommenders on the strength of that evidence. However, attention models and SSMs differ in their inductive biases, and whether the benefits of CL transfer between them has not been investigated. 

This study evaluates data-level contrastive learning, following CL4SRec, on two SSM backbones, Mamba4Rec and SIGMA, using InfoNCE and Decoupled Contrastive Learning (DCL) with Cross-Entropy and Bayesian Personalized Ranking as primary losses. Across four benchmark datasets and 60 experiments, accuracy (Hit@K, NDCG@K, MRR@K), representation geometry (isotropy, effective rank), item coverage, and training behaviour were compared against non-contrastive baselines and attention-based models (SASRec, CL4SRec, DuoRec). 

### Key Findings
*   On average, data-level CL degraded SSM accuracy, representation geometry, and item coverage, whereas CL4SRec improved on SASRec in accuracy and isotropy.
*   Preliminary model-level experiments, trained for only 10 epochs, avoided this geometric collapse but did not produce consistent accuracy gains. 
*   These results suggest that data-level CL does not transfer from attention-based to SSM-based recommenders, and that contrastive regularisation may need to be adapted to a model’s inductive bias.

---
*Keywords: Contrastive Learning, State Space Models, Sequential Recommendation, Representation Learning*
