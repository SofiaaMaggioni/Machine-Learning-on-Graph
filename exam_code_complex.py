import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. SETUP SPERIMENTALE (Grafo di Esempio)
# ==========================================
# Creiamo un grafo sintetico (100 nodi) per la rete di coautoria passata
G_past = nx.erdos_renyi_graph(n=100, p=0.05, seed=42)

# Simuliamo i link futuri (Ground Truth) basati su logiche di prossimità globale
G_future_links = []
np.random.seed(42)
for u, v in nx.non_edges(G_past):
    if nx.has_path(G_past, u, v):
        shortest_path_len = nx.shortest_path_length(G_past, source=u, target=v) if nx.has_path(G_past, u, v) else 10
        deg_prod = G_past.degree(u) * G_past.degree(v)
        
        # Link favoriti da cammini corti (Katz) o nodi centrali (PageRank)
        if shortest_path_len == 2 and np.random.rand() > 0.4:
            G_future_links.append((u, v))
        elif shortest_path_len == 3 and deg_prod > 30 and np.random.rand() > 0.6:
            G_future_links.append((u, v))

print(f"Link nel passato (G_past): {G_past.number_of_edges()}")
print(f"Nuovi link formatisi nel futuro: {len(G_future_links)}")

# Pre-calcoliamo la centralità di Katz globale per supportare l'estrazione successiva
# alpha deve essere inferiore al reciproco del raggio spettrale per garantire la convergenza
import scipy as sp
try:
    L = nx.adjacency_matrix(G_past).astype(float)
    eigenvalues = sp.sparse.linalg.eigs(L, k=1, which='LM', return_eigenvectors=False)
    max_eigen = np.abs(eigenvalues[0])
    katz_alpha = 0.85 / max_eigen  # Condizione di convergenza
except:
    katz_alpha = 0.05

# Calcoliamo l'intero dizionario di Katz Centrality sulla rete passata
katz_dict = nx.katz_centrality(G_past, alpha=katz_alpha, max_iter=1000)

# ==========================================
# 2. FEATURE ENGINEERING AVANZATO
# ==========================================
def extract_advanced_features(graph, edge_list, label, katz_centrality_dict):
    features = []
    
    # Per ottimizzare il Rooted PageRank evitiamo di ricalcolare l'intero grafo per ogni coppia.
    # Calcoliamo il Personalized PageRank nodo per nodo quando necessario.
    memo_pagerank = {}

    for u, v in edge_list:
        # A. Common Neighbors (Locale)
        common_neigh = len(list(nx.common_neighbors(graph, u, v)))

        # B. Jaccard Coefficient
        jaccard = list(nx.jaccard_coefficient(graph, [(u, v)]))[0][2]
        
        # c. Adamic/Adar Index (Locale con pesi)
        try:
            adamic_adar = list(nx.adamic_adar_index(graph, [(u, v)]))[0][2]
        except ZeroDivisionError:
            adamic_adar = 0.0
        
        # D. Preferential Attachment
        pref_attach = list(nx.preferential_attachment(graph, [(u, v)]))[0][2]
        
            
        # E. Katz Index (Globale - combinazione delle centralità dei due nodi d'origine)
        # Sfrutta la centralità dei nodi come proxy dell'intensità dei cammini infiniti tra loro
        katz_score = katz_centrality_dict[u] * katz_centrality_dict[v]
        
        # F. Rooted PageRank (Stocastico - Walk con reset su nodo sorgente)
        if u not in memo_pagerank:
            # Alpha=0.85 corrisponde alla probabilità di jump (1 - damping factor) del paper
            memo_pagerank[u] = nx.pagerank(graph, alpha=0.85, personalization={u: 1})
        
        rooted_pr_score = memo_pagerank[u].get(v, 0.0)
        
        features.append({
            'node_u': u,
            'node_v': v,
            'common_neighbors': common_neigh,
            'jaccard': jaccard,
            'adamic_adar': adamic_adar,
            'preferential_attachment': pref_attach,
            'katz_index': katz_score,
            'rooted_pagerank': rooted_pr_score,
            'label': label
        })
    return pd.DataFrame(features)

print("\nEstrazione feature per i link positivi...")
df_positive = extract_advanced_features(G_past, G_future_links, label=1, katz_centrality_dict=katz_dict)

print("Estrazione feature per i link negativi (Edge Masking)...")
all_non_edges = list(nx.non_edges(G_past))
negative_candidates = [edge for edge in all_non_edges if edge not in G_future_links]
np.random.shuffle(negative_candidates)
negative_samples = negative_candidates[:len(G_future_links)*4] # Campionamento bilanciato/balanced sampling len*4 si sbilancia

df_negative = extract_advanced_features(G_past, negative_samples, label=0, katz_centrality_dict=katz_dict)

# Unione dei due campioni nel dataset finale
df_dataset = pd.concat([df_positive, df_negative], axis=0).reset_index(drop=True)

# ==========================================
# 3. PIPELINE DI MACHINE LEARNING
# ==========================================
feature_cols = ['common_neighbors', 'jaccard', 'adamic_adar', 'preferential_attachment', 'katz_index', 'rooted_pagerank']
X = df_dataset[feature_cols]
y = df_dataset['label']

# Split in Train e Test (Stratificato per mantenere le proporzioni delle classi)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Z-score Normalization (Standardizzazione necessaria data l'eterogeneità delle metriche)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Addestramento della Regressione Logistica Multivariata
clf = LogisticRegression()
clf.fit(X_train_scaled, y_train)

# ==========================================
# 4. VALUTAZIONE DELLE PERFORMANCE
# ==========================================
y_pred = clf.predict(X_test_scaled)
y_prob = clf.predict_proba(X_test_scaled)[:, 1]

print("\n" + "="*40)
print("--- REPORT DI CLASSIFICAZIONE AVANZATO ---")
print("="*40)
print(classification_report(y_test, y_pred))
print(f"ROC AUC Score finale: {roc_auc_score(y_test, y_prob):.4f}")

print("\n--- COEFFICIENTI DELLE METRICHE GLOBALI E LOCALI ---")
for feature, coef in zip(X.columns, clf.coef_[0]):
    print(f"{feature}: {coef:.4f}")



import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 1. Estraiamo le probabilità dal TUO modello (che nel codice si chiama 'lr')
# Usiamo 'X_test_scaled' perché le feature del test set nel tuo script sono normalizzate
y_probs = clf.predict_proba(X_test_scaled)[:, 1]

# 2. Calcolo dei punti della curva ROC e dell'area AUC
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# 3. Configurazione del grafico per l'esame
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'Modello Supervisionato (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Classificatore Casuale (AUC = 0.500)')

# Formattazione degli assi e titoli
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate (FPR)', fontsize=11, fontweight='bold', labelpad=10)
plt.ylabel('True Positive Rate (TPR)', fontsize=11, fontweight='bold', labelpad=10)
plt.title('Curva ROC - Link Prediction Pipeline', fontsize=13, fontweight='bold', pad=15)
plt.legend(loc="lower right", fontsize=10, frameon=True)
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()

# 4. Salvataggio dell'immagine finale da mostrare al prof
plt.savefig('curva_roc_link_prediction.png', dpi=300)
plt.show()

print(f"\n[OK] Grafico salvato come 'curva_roc_link_prediction.png' con AUC: {roc_auc:.4f}")