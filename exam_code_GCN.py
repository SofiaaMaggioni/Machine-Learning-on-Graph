import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report

# Controllo se PyTorch Geometric è installato per guidare l'utente
try:
    from torch_geometric.nn import GCNConv
    from torch_geometric.utils import from_networkx
except ImportError:
    raise ImportError("Per eseguire questo codice devi installare PyTorch Geometric. "
                      "Esegui: pip install torch torch-geometric")

# =========================================================================
# PASSO 1: SETUP TEMPORALE E SIMULAZIONE DATASET (Inalterato dal paper)
# =========================================================================
print("Fase 1: Inizializzazione dei grafi temporali...")
# Generiamo un grafo sintetico ad hoc (es. Barabási-Albert) per simulare arXiv
G_full = nx.barabasi_albert_graph(n=500, m=3, seed=42)

# Definiamo il Core (Nodi attivi con grado >= 3)
core_nodes = [node for node, degree in G_full.degree() if degree >= 3]
G_full = G_full.subgraph(core_nodes).copy()

# Dividiamo gli archi in Passato (Training) e Futuro (Test)
all_edges = list(G_full.edges())
np.random.seed(42)
np.random.shuffle(all_edges)

split_idx = int(len(all_edges) * 0.6)
edges_past = all_edges[:split_idx]
edges_future = all_edges[split_idx:]

# Creiamo il grafo del passato G_past (Gcollab del paper)
G_past = nx.Graph()
G_past.add_nodes_from(G_full.nodes())
G_past.add_edges_from(edges_past)

# Aggiungiamo i self-loops a G_past come spiegato nella Lezione 10
G_past_with_loops = G_past.copy()
G_past_with_loops.add_edges_from([(n, n) for n in G_past.nodes()])

print(f"Link nel passato (G_past): {G_past.number_of_edges()}")
print(f"Nuovi link reali nel futuro: {len(edges_future)}")

# =========================================================================
# PASSO 2: PREPARAZIONE DEI DATI PER PYTORCH GEOMETRIC 
# =========================================================================
# Convertiamo il grafo di NetworkX (con self-loops) in un oggetto PyG Data
pyg_data = from_networkx(G_past_with_loops)

# Definizione Feature Iniziali: non avendo attributi, usiamo un vettore di "1" per ogni nodo (Lezione 10)
num_nodes = G_past.number_of_nodes()
num_features = 32  # Dimensione della feature di base (inizializzata a 1)
pyg_data.x = torch.ones((num_nodes, num_features), dtype=torch.float)

# Creazione dei campioni per l'addestramento del Link Prediction Decoder
# Link Positivi (classe 1): Archi che si formano nel futuro
pos_edge_index = torch.tensor(edges_future, dtype=torch.long).t()

# Link Negativi (classe 0): Coppie di nodi che non si connettono né nel passato né nel futuro
neg_edges = []
nodes_list = list(G_past.nodes())
while len(neg_edges) < len(edges_future):
    u, v = np.random.choice(nodes_list, size=2, replace=False)
    if not G_past.has_edge(u, v) and (u, v) not in edges_future and (v, u) not in edges_future:
        neg_edges.append((u, v))
neg_edge_index = torch.tensor(neg_edges, dtype=torch.long).t()

# =========================================================================
# PASSO 3: ARCHITETTURA GCN + LINK DECODER (Lezione 7 & 10)
# =========================================================================
class GCNLinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNLinkPredictor, self).__init__()
        # Definizione dei due layer convolutivi che applicano la normalizzazione simmetrica D^-1/2 A D^-1/2
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        # Primo layer di message passing + Attivazione non lineare (ReLU)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # Secondo layer: genera gli embeddings finali Z
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_index):
        # Decoder: Prodotto scalare tra i vettori dei nodi sorgente e target (Lezione 7)
        src = z[edge_index[0]]
        tgt = z[edge_index[1]]
        # Calcolo dell'inner product riga per riga
        scores = torch.sum(src * tgt, dim=-1)
        return torch.sigmoid(scores)

# Istanziamo il modello (embedding finale a 16 dimensioni)
model = GCNLinkPredictor(in_channels=num_features, hidden_channels=64, out_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

# =========================================================================
# PASSO 4: ADDESTRAMENTO END-TO-END (Optimization Loop)
# =========================================================================
print("\nInizio addestramento della Graph Convolutional Network...")
model.train()
for epoch in range(1, 151):
    optimizer.zero_grad()
    
    # 1. Fase di Encoding: La GCN genera la matrice degli embeddings Z
    z = model.encode(pyg_data.x, pyg_data.edge_index)
    
    # 2. Fase di Decoding: Calcolo degli score per link positivi e negativi
    pos_pred = model.decode(z, pos_edge_index)
    neg_pred = model.decode(z, neg_edge_index)
    
    # Concateneremo le predizioni e calcoleremo la loss rispetto alla Ground Truth
    predictions = torch.cat([pos_pred, neg_pred])
    targets = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))])
    
    loss = criterion(predictions, targets)
    loss.backward()
    optimizer.step()
    
    if epoch % 30 == 0:
        print(f"Epoca {epoch:03d} | Loss: {loss.item():.4f}")

# =========================================================================
# PASSO 5: VALUTAZIONE METRICHE DI PERFORMANCE
# =========================================================================
model.eval()
with torch.no_grad():
    z = model.encode(pyg_data.x, pyg_data.edge_index)
    pos_pred = model.decode(z, pos_edge_index).numpy()
    neg_pred = model.decode(z, neg_edge_index).numpy()

y_true = np.concatenate([np.ones(len(pos_pred)), np.zeros(len(neg_pred))])
y_pred_scores = np.concatenate([pos_pred, neg_pred])
y_pred_labels = np.where(y_pred_scores >= 0.5, 1, 0)

print("\n" + "="*40)
print("--- REPORT DI CLASSIFICAZIONE GEOMETRICO (GCN) ---")
print(classification_report(y_true, y_pred_labels))
print(f"ROC AUC Score finale: {roc_auc_score(y_true, y_pred_scores):.4f}")
print("="*40)



import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Supponendo che 'model' sia la tua GCN addestrata e 'x', 'edge_index' siano i tuoi dati PyTorch Geometric
model.eval()
with torch.no_grad():
    # Estraiamo gli embedding prima del decoder dei link
    embeddings = model.encode(pyg_data.x, pyg_data.edge_index).cpu().numpy() 

# Riduciamo le dimensioni a 2D con t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Calcoliamo il grado dei nodi per usarlo come colore del grafico
# (mostra come la GCN raggruppa nodi con ruoli strutturali simili)
degrees = [G_past.degree(i) for i in range(G_past.number_of_nodes())]

plt.figure(figsize=(8, 6))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=degrees, cmap='viridis', s=50)
plt.colorbar(scatter, label='Grado del Nodo')
plt.title('Visualizzazione 2D degli Embedding dei Nodi tramite t-SNE', fontsize=12, fontweight='bold')
plt.xlabel('Dimensione t-SNE 1')
plt.ylabel('Dimensione t-SNE 2')
plt.grid(True, linestyle=':', alpha=0.5)
plt.show()