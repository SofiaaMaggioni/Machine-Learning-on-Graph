# Machine learning on graph

This project investigates and empirically extends the seminal study by **Liben-Nowell and Kleinberg (2007)** on the **Link Prediction** problem in complex networks, using temporal co-authorship graphs extracted from **arXiv**. The primary objective is to evaluate whether and how future scientific collaborations can be predicted purely from network topology, without relying on external node attributes.

The repository contrasts and evaluates two foundational paradigms:

1. **Supervised Feature Engineering (Classical Pipeline):**
   - Transcends the original paper's univariate heuristic ranking by aggregating multiple topological proximity measures into a single feature matrix.
   - **Extracted Features**: Local neighborhood metrics (*Common Neighbors*, *Jaccard Coefficient*, *Adamic/Adar*), growth models (*Preferential Attachment*), and global/random-walk measures (*Katz Index*, *Rooted PageRank*).
   - **Experimental Rigor**: Strictly enforced temporal split between G_past and G_future to prevent data leakage, combined with balanced **Negative Sampling** (50% positive links, 50% non-links).
   - **Model & Performance**: A standardized **Logistic Regression** classifier achieves a **ROC AUC of ~0.95**, ensuring transparent interpretability through feature weight analysis (e.g., assessing collinearity between degree-based and rarity-penalized metrics).

2. **Geometric Deep Learning (GCN Pipeline):**
   - Implements a modern **Graph Representation Learning** pipeline using a 2-layer **Graph Convolutional Network (GCN)** in **PyTorch Geometric**.
   - **Uninformative Initialization**: Nodes are initialized with constant all-ones feature vectors, forcing the network to extract representations solely via topological *message passing*.
   - **Encoder-Decoder Framework**: A 2-hop GCN encoder maps nodes into a 16-dimensional continuous latent space, while a dot-product link prediction head calculates edge probabilities.
   - **Latent Space Inspection**: The learned embeddings are projected via **t-SNE**, visually confirming the emergence of spontaneous community clusters and structural roles without manual feature extraction.

##
Grade: 25

AY: 2025-2026
