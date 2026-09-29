import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from scipy.optimize import differential_evolution

from qiskit import QuantumCircuit, transpile
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
try:
    from iqm.qiskit import IQMProvider
except ImportError:
    from iqm.qiskit_iqm import IQMProvider

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, matthews_corrcoef

import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. PARSOWANIE TOP 5 Z RANKINGU HUBÓW
# ==========================================
csv_results_path = "wyniki_wszystkie_huby_posortowane.csv"
if not os.path.exists(csv_results_path):
    raise FileNotFoundError(f"Brak pliku {csv_results_path}. Upewnij się, że jest w tym samym folderze.")

df_top = pd.read_csv(csv_results_path)
# Pobieramy rygorystycznie Top 5 najlepszych wierszy z pliku
top_n = 5
candidates = df_top.head(top_n)

bank_mapping = {'PKO': 'pbv_pko', 'PEKAO': 'pbv_peo', 'SAN': 'pbv_san', 'ING': 'pbv_ing', 'MBK': 'pbv_mbk'}

def extract_config(row):
    feat_map = []
    hub_idx = -1
    cols = ['PKO (q0)', 'PEKAO (q1)', 'SAN (q2)', 'ING (q3)', 'MBK (q4)']
    for q, col in enumerate(cols):
        val = str(row[col])
        if "[HUB]" in val:
            hub_idx = q
            val = val.replace("[HUB]", "").strip()
        parts = val.split("+")
        b = parts[0].strip()
        m = parts[1].strip()
        feat_map.extend([bank_mapping[b], m])
    return feat_map, hub_idx

# Wczytanie głównej bazy danych
csv_path = "dataset_final.csv" if os.path.exists("dataset_final.csv") else "dane/dataset_final.csv"
df = pd.read_csv(csv_path, index_col=0)
df.index = pd.to_datetime(df.index)
df = df.sort_index()

y_raw = df["target_y"].astype(int).values
dates_raw = df.index.values
LOOKBACK = 52

valid_indices = [t for t in range(LOOKBACK, len(y_raw))]
y = y_raw[valid_indices]
dates = dates_raw[valid_indices]

# ==========================================
# 2. FUNKCJE KWANTOWE
# ==========================================
def build_base_ansatz(x, num_qubits=5, hub_qubit=1):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(x[2 * i], i)
        qc.rz(x[2 * i + 1], i)
    for i in range(num_qubits):
        if i != hub_qubit:
            qc.cx(hub_qubit, i)
    return qc

def build_measurement_circuits(x, num_qubits=5, hub_qubit=1):
    qc_x = build_base_ansatz(x, num_qubits, hub_qubit)
    for i in range(num_qubits): qc_x.h(i)
    qc_x.measure_all()
    
    qc_y = build_base_ansatz(x, num_qubits, hub_qubit)
    for i in range(num_qubits): qc_y.sdg(i); qc_y.h(i)
    qc_y.measure_all()
    
    qc_z = build_base_ansatz(x, num_qubits, hub_qubit)
    qc_z.measure_all()
    return qc_x, qc_y, qc_z

def get_fast_sim_projections(X_q, hub_qubit, num_qubits=5):
    estimator = StatevectorEstimator()
    observables = [SparsePauliOp("".join(['I' if j != q else p for j in reversed(range(num_qubits))])) 
                   for q in range(num_qubits) for p in ['X', 'Y', 'Z']]
    Phi = np.zeros((len(X_q), num_qubits * 3))
    circuits = [build_base_ansatz(x, num_qubits, hub_qubit) for x in X_q]
    job = estimator.run([(circ, observables) for circ in circuits])
    for i, res in enumerate(job.result()): Phi[i] = res.data.evs
    return Phi

def find_best_weights(X_base, y_true, hub_qubit):
    split_idx = len(X_base) // 2
    X_train, y_train = X_base[:split_idx], y_true[:split_idx]
    
    def objective_function(weights):
        w_map = {}
        w_idx = 0
        for q in range(5):
            if q == hub_qubit: w_map[q] = 1.0
            else: w_map[q] = weights[w_idx]; w_idx += 1
            
        multipliers = np.zeros(10)
        for q in range(5): multipliers[2*q] = w_map[q]; multipliers[2*q+1] = w_map[q]
        
        X_weighted = X_train * multipliers
        Phi_sim = get_fast_sim_projections(X_weighted, hub_qubit)
        
        inner_cv = TimeSeriesSplit(n_splits=3)
        pipeline = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", probability=True))])
        try:
            search = GridSearchCV(pipeline, {"svm__C": [0.1, 1.0, 5.0], "svm__gamma": ["scale"]}, 
                                  cv=inner_cv, scoring="roc_auc", n_jobs=-1).fit(Phi_sim, y_train)
            return -search.best_score_ 
        except: return 0.0

    bounds = [(0.1, 1.0)] * 4
    result = differential_evolution(objective_function, bounds, maxiter=5, popsize=4, seed=42, disp=False)
    
    final_weights = np.ones(5)
    w_idx = 0
    for q in range(5):
        if q != hub_qubit: final_weights[q] = result.x[w_idx]; w_idx += 1
    return final_weights

def get_hw_layout(hub_logical):
    layout = [-1]*5
    layout[hub_logical] = 2
    phys = [0, 1, 3, 4]
    idx = 0
    for i in range(5):
        if i != hub_logical:
            layout[i] = phys[idx]
            idx += 1
    return layout

def expectation_from_counts(counts, num_qubits):
    total_shots = sum(counts.values())
    expvals = np.zeros(num_qubits)
    for bitstring, count in counts.items():
        bits = [1 if b == '0' else -1 for b in reversed(bitstring.replace(" ", ""))]
        for q in range(num_qubits): expvals[q] += bits[q] * count
    return expvals / total_shots

def get_physical_qpu_projections_cached(X_q, weights_arr, name_prefix, hub_qubit, num_qubits=5, shots=4000):
    w_str = "_".join([f"{w:.2f}" for w in weights_arr])
    cache_file_npz = f"phi_odra_{name_prefix}_q{hub_qubit}_{w_str}.npz"
    cache_file_csv = f"phi_odra_{name_prefix}_q{hub_qubit}_{w_str}.csv"
    
    if os.path.exists(cache_file_npz):
        print(f"  [QPU] Wczytano {name_prefix} z pliku cache: {cache_file_npz}")
        return np.load(cache_file_npz)["phi"]

    print(f"  [QPU] Łączenie z maszyną Odra dla {name_prefix} (HUB q{hub_qubit})...")
    env_paths = ["../iqm_token.env", "iqm_token.env", "token.env", "../token.env"]
    for path in env_paths:
        if os.path.exists(path): load_dotenv(path); break
            
    server_url = os.getenv("SERVER")
    if not server_url: raise ValueError("Brak adresu SERVER w pliku .env.")
    
    provider = IQMProvider(server_url)
    backend = provider.get_backend()
    
    all_circuits = []
    for x in X_q: all_circuits.extend(build_measurement_circuits(x, num_qubits, hub_qubit))
    
    hw_layout = get_hw_layout(hub_qubit) 
    transpiled = transpile(all_circuits, backend=backend, optimization_level=2, initial_layout=hw_layout)
    
    all_counts = []
    batch_size = 50
    for i in range(0, len(transpiled), batch_size):
        batch = transpiled[i : i + batch_size]
        print(f"      Wysyłanie paczki {i//batch_size + 1} / {(len(transpiled)-1)//batch_size + 1}...")
        job = backend.run(batch, shots=shots)
        for j in range(len(batch)): all_counts.append(job.result().get_counts(j))
            
    Phi_qpu = np.zeros((len(X_q), num_qubits * 3))
    for i in range(len(X_q)):
        for q in range(num_qubits):
            Phi_qpu[i, 3*q+0] = expectation_from_counts(all_counts[3*i], num_qubits)[q]
            Phi_qpu[i, 3*q+1] = expectation_from_counts(all_counts[3*i+1], num_qubits)[q]
            Phi_qpu[i, 3*q+2] = expectation_from_counts(all_counts[3*i+2], num_qubits)[q]
            
    np.savez(cache_file_npz, phi=Phi_qpu)
    pd.DataFrame(Phi_qpu).to_csv(cache_file_csv, index=False)
    print(f"  [QPU] WYPLUTO DANE: Zapisano rzuty do {cache_file_npz} oraz {cache_file_csv}")
    
    return Phi_qpu

# ==========================================
# 3. PĘTLA PO TOP 5 KANDYDATACH Z OSOBNĄ OPTYMALIZACJĄ WAG
# ==========================================
MODELS = {}

for iteration, (idx, row) in enumerate(candidates.iterrows()):
    print(f"\n{'='*60}\n>>> ANALIZA KANDYDATA NR {iteration+1} z TOP 5 (Rank Index {idx})\n{'='*60}")
    feat_map, hub_q = extract_config(row)
    print(f"HUB logiczny: q{hub_q} | Cechy: {feat_map}")
    
    # Skalowanie przyczynowe dla konkretnego kandydata
    X_raw_cand = df[feat_map].values
    X_norm_list = []
    for t in range(LOOKBACK, len(X_raw_cand)):
        past_window = X_raw_cand[t - LOOKBACK : t]
        min_vals = past_window.min(axis=0)
        max_vals = past_window.max(axis=0)
        range_vals = np.where((max_vals - min_vals) == 0, 1e-8, max_vals - min_vals)
        scaled = ((X_raw_cand[t] - min_vals) / range_vals) * (2 * np.pi)
        X_norm_list.append(np.clip(scaled, 0.0, 2 * np.pi))
        
    X_quantum_base = np.array(X_norm_list)
    if iteration == 0:
        MODELS["Klasyczny SVM (Raw Data)"] = X_raw_cand[valid_indices]
        
    # Osobna optymalizacja wag dla tego konkretnego kandydata
    print(f" -> Optymalizacja wag dla kandydata {idx}...")
    best_w = find_best_weights(X_quantum_base, y, hub_q)
    print(f" -> Zoptymalizowane wagi: {[round(w, 2) for w in best_w]}")
    
    # Generowanie rzutów QPU (Baseline: równe wagi)
    w_base = [1.0] * 5
    phi_base = get_physical_qpu_projections_cached(X_quantum_base, w_base, f"Top5_Rank{idx}_Base", hub_q)
    MODELS[f"[Top5 #{iteration+1}] Kwantowy (Baseline)"] = phi_base
    
    # Generowanie rzutów QPU (Zoptymalizowane wagi osobno dla tego kandydata)
    multipliers = np.zeros(10)
    for q in range(5): multipliers[2*q] = best_w[q]; multipliers[2*q+1] = best_w[q]
    X_weighted = X_quantum_base * multipliers
    
    phi_opt = get_physical_qpu_projections_cached(X_weighted, best_w, f"Top5_Rank{idx}_Opt", hub_q)
    MODELS[f"[Top5 #{iteration+1}] Kwantowy (Zoptymalizowany)"] = phi_opt

# ==========================================
# 4. TURNIEJ WALK-FORWARD I WYNIKI
# ==========================================
print("\n>>> ROZPOCZYNAM WIELKI TURNIEJ OOS (Walk-Forward dla Top 5) <<<")

tournament_results = {}
TRAIN_WINDOW = 52
pipeline = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", probability=True))])
inner_cv = TimeSeriesSplit(n_splits=3)
param_grid = {"svm__C": [0.1, 1.0, 5.0, 10.0], "svm__gamma": ["scale", "auto"]}

for name, X_eval in MODELS.items():
    print(f"[Ewaluacja] {name}")
    y_true_final, probs = [], []
    for t in range(TRAIN_WINDOW, len(y)):
        X_tr, X_te = X_eval[t - TRAIN_WINDOW : t], X_eval[t : t + 1]
        y_tr = y[t - TRAIN_WINDOW : t]
        
        search = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring="roc_auc", n_jobs=-1).fit(X_tr, y_tr)
        probs.append(search.predict_proba(X_te)[0, 1])
        y_true_final.append(y[t])
        
    tournament_results[name] = {"probs": np.array(probs), "y_true": np.array(y_true_final)}

df_metrics = pd.DataFrame(index=MODELS.keys(), columns=["Accuracy (Thresh 0.55)", "AUC ROC", "MCC"])
DECISION_THRESHOLD = 0.55

for name in MODELS.keys():
    y_t = tournament_results[name]["y_true"]
    p_t = tournament_results[name]["probs"]
    pred = (p_t > DECISION_THRESHOLD).astype(int)
    
    df_metrics.loc[name] = [accuracy_score(y_t, pred), roc_auc_score(y_t, p_t), matthews_corrcoef(y_t, pred)]

print("\n" + "="*75)
print("--- OSTATECZNE WYNIKI TURNIEJU TOP 5 KANDYDATÓW (PRÓG 0.55) ---")
print("="*75)
print(df_metrics.to_string(float_format="{:.4f}".format))
print("="*75)

csv_out = "tabela_wynikow_top5_kandydaci.csv"
df_metrics.to_csv(csv_out, float_format="%.4f")
print(f"\n>>> Zapisano tabelę z wynikami do pliku: {csv_out}")

# ==========================================
# 5. WIZUALIZACJA I WYŚWIETLENIE WYKRESU
# ==========================================
fig, ax = plt.subplots(1, 2, figsize=(18, 6))

color_palette = ['#e74c3c', '#2980b9', '#27ae60', '#8e44ad', '#d35400'] # Paleta dla 5 kandydatów

for name in MODELS.keys():
    y_t = tournament_results[name]["y_true"]
    p_t = tournament_results[name]["probs"]
    fpr, tpr, _ = roc_curve(y_t, p_t)
    
    net_hits = np.cumsum(np.where((p_t > DECISION_THRESHOLD).astype(int) == y_t, 1, -1))
    
    if "Klasyczny" in name:
        ax[0].plot(fpr, tpr, color='#7f8c8d', lw=2, ls='--', label=f'{name} (AUC = {df_metrics.loc[name, "AUC ROC"]:.3f})')
        ax[1].plot(dates[TRAIN_WINDOW:], net_hits, color='#7f8c8d', lw=2, ls='--', label=name)
    else:
        # Wyciąganie numeru kandydata z nazwy
        cand_str = name.split("]")[0].replace("[Top5 #", "")
        cand_idx = int(cand_str) - 1
        color = color_palette[cand_idx % len(color_palette)]
        
        lw, ls = (2, ':') if "Baseline" in name else (3, '-')
        ax[0].plot(fpr, tpr, color=color, lw=lw, ls=ls, label=f'{name} (AUC = {df_metrics.loc[name, "AUC ROC"]:.3f})')
        ax[1].plot(dates[TRAIN_WINDOW:], net_hits, color=color, lw=lw, ls=ls, label=name)

ax[0].plot([0, 1], [0, 1], 'k:', alpha=0.5)
ax[0].set_title('Krzywa ROC - Top 5 Konfiguracji z Osobną Optymalizacją Wag', fontweight='bold')
ax[0].set_xlabel('False Positive Rate')
ax[0].set_ylabel('True Positive Rate')
ax[0].legend(fontsize='x-small', loc='lower right')
ax[0].grid(True, linestyle='--', alpha=0.6)

ax[1].axhline(0, color='black', lw=1.5)
ax[1].set_title('Skumulowany Bilans Trafień (Próg 0.55)', fontweight='bold')
ax[1].set_xlabel('Data')
ax[1].set_ylabel('Net Hits')
ax[1].legend(fontsize='x-small', loc='upper left')
ax[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("wykres_turniej_top5_kandydatow.png", dpi=300, bbox_inches='tight')
print(">>> Zapisano wykres do pliku: wykres_turniej_top5_kandydatow.png")

plt.show()