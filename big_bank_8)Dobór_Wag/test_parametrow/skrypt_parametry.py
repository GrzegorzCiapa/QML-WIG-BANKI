import os
import itertools
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances
import warnings

warnings.filterwarnings("ignore")

csv_path = "dataset_final.csv" if os.path.exists("dataset_final.csv") else "dane/dataset_final.csv"

BANK_NAMES = ['PKO', 'PEKAO', 'SAN', 'ING', 'MBK']
BANKS = ['pbv_pko', 'pbv_peo', 'pbv_san', 'pbv_ing', 'pbv_mbk']
MACRO = ['wibor_3m', 'rentownosc_10y', 'eurpln', 'wig20', 'zmiennosc']

ALL_FEATURES = BANKS + MACRO
df = pd.read_csv(csv_path, index_col=0)
df.index = pd.to_datetime(df.index)
df = df.sort_index()

X_raw = df[ALL_FEATURES].values
y_raw = df["target_y"].astype(int).values

LOOKBACK = 52
X_norm_list, valid_indices = [], []
for t in range(LOOKBACK, len(X_raw)):
    past_window = X_raw[t - LOOKBACK : t]
    min_vals = past_window.min(axis=0)
    max_vals = past_window.max(axis=0)
    range_vals = np.where((max_vals - min_vals) == 0, 1e-8, max_vals - min_vals)
    scaled = np.clip(((X_raw[t] - min_vals) / range_vals) * (2 * np.pi), 0.0, 2 * np.pi)
    X_norm_list.append(scaled)
    valid_indices.append(t)

X_norm = np.array(X_norm_list)
y = y_raw[valid_indices]

X_banks = X_norm[:, :5]  
X_macro = X_norm[:, 5:]  

NUM_QUBITS = 5

def build_base_ansatz(x_b, x_m, num_qubits, current_hub):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(x_b[i], i)
        qc.rz(x_m[i], i)
    for i in range(num_qubits):
        if i != current_hub:
            qc.cx(current_hub, i)
    return qc

estimator = StatevectorEstimator()
observables = [SparsePauliOp("".join(['I' if j != q else p for j in reversed(range(NUM_QUBITS))])) 
               for q in range(NUM_QUBITS) for p in ['X', 'Y', 'Z']]

macro_permutations = list(itertools.permutations(range(5)))
results_list = []
TRAIN_WINDOW = 52
n_samples = len(y)

print(f">>> Testowanie wszystkich 600 powiązań (5 opcji HUB x 120 permutacji Makro)...")

for hub_idx in range(NUM_QUBITS):
    hub_bank_name = BANK_NAMES[hub_idx]
    print(f"\n--- Rozpoczynam obliczenia dla HUB: {hub_bank_name} (q{hub_idx}) ---")
    
    for idx, m_perm in enumerate(macro_permutations):
        X_m_perm = X_macro[:, m_perm]
        
        circuits = [build_base_ansatz(X_banks[i], X_m_perm[i], NUM_QUBITS, hub_idx) for i in range(n_samples)]
        job = estimator.run([(qc, observables) for qc in circuits])
        Phi = np.array([res.data.evs for res in job.result()])

        y_true_out, y_pred_out, p_pred_out = [], [], []
        for t in range(TRAIN_WINDOW, n_samples):
            P_tr, P_te = Phi[t - TRAIN_WINDOW : t], Phi[t : t + 1]
            y_tr, y_te = y[t - TRAIN_WINDOW : t], y[t]
            y_true_out.append(y_te)
            
            if len(np.unique(y_tr)) > 1:
                dists = pairwise_distances(P_tr)
                non_zero_dists = dists[dists > 0]
                
                gamma_val = 1.0 if len(non_zero_dists) == 0 else 1.0 / (2.0 * (np.median(non_zero_dists) ** 2) + 1e-8)
                    
                K_tr, K_te = rbf_kernel(P_tr, P_tr, gamma=gamma_val), rbf_kernel(P_te, P_tr, gamma=gamma_val)
                
                clf = SVC(kernel="precomputed", probability=True, C=5.0).fit(K_tr, y_tr)
                p_tr_pred = clf.predict_proba(K_tr)[:, 1]
                p_te_pred = clf.predict_proba(K_te)[0, 1]
                p_pred_out.append(p_te_pred)
                
                best_thresh, best_f1 = 0.5, 0.0
                for thresh in np.linspace(0.35, 0.65, 31):
                    f1 = f1_score(y_tr, (p_tr_pred >= thresh).astype(int))
                    if f1 > best_f1: 
                        best_f1, best_thresh = f1, thresh
                
                y_pred_out.append(1 if p_te_pred >= best_thresh else 0)
            else:
                p_pred_out.append(y_tr[0])
                y_pred_out.append(y_tr[0])

        auc = roc_auc_score(y_true_out, p_pred_out)
        acc = accuracy_score(y_true_out, y_pred_out)
        f1_final = f1_score(y_true_out, y_pred_out)
        
        results_list.append({
            "Aktywny_HUB": f"q{hub_idx} ({hub_bank_name})",
            "Perm_Makro": str(m_perm),
            "PKO (q0)": f"PKO + {MACRO[m_perm[0]]}" + (" [HUB]" if hub_idx == 0 else ""),
            "PEKAO (q1)": f"PEKAO + {MACRO[m_perm[1]]}" + (" [HUB]" if hub_idx == 1 else ""),
            "SAN (q2)": f"SAN + {MACRO[m_perm[2]]}" + (" [HUB]" if hub_idx == 2 else ""),
            "ING (q3)": f"ING + {MACRO[m_perm[3]]}" + (" [HUB]" if hub_idx == 3 else ""),
            "MBK (q4)": f"MBK + {MACRO[m_perm[4]]}" + (" [HUB]" if hub_idx == 4 else ""),
            "AUC ROC": auc,
            "Accuracy": acc,
            "F1-Score": f1_final
        })

        if (idx + 1) % 10 == 0 or idx == 119:
            pd.DataFrame(results_list).to_csv("wyniki_wszystkie_huby.csv", index=False)
            
        print(f"[{hub_bank_name} HUB] Permutacja {idx+1}/120 | AUC: {auc:.4f}", flush=True)

df_res = pd.DataFrame(results_list).sort_values(by="AUC ROC", ascending=False)
df_res.to_csv("wyniki_wszystkie_huby_posortowane.csv", index=False)
print(">>> Zakończono! Wyniki posortowane zapisano do: wyniki_wszystkie_huby_posortowane.csv")