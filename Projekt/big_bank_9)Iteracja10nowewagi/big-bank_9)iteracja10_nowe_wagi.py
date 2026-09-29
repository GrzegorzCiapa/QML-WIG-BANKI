import os
import sys
import time
import json
import datetime
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# IMPORTY
# ==========================================
from qiskit import QuantumCircuit, transpile
from iqm.qiskit_iqm import IQMProvider
from dotenv import load_dotenv

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef

import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 0. WSZYSTKIE ŚCIEŻKI WZGLĘDEM LOKALIZACJI SKRYPTU
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def sp(*parts):
    """Ścieżka względem folderu skryptu."""
    return os.path.join(SCRIPT_DIR, *parts)

print(f">>> Folder skryptu (tu trafią wszystkie wyniki i cache): {SCRIPT_DIR}")

# ==========================================
# 1. KONFIGURACJA TESTU (FIZYCZNA ODRA)
# ==========================================
NUM_RUNS = 10
TRAIN_WINDOW = 52

# --- ZWYCIĘSKA, WOLNA OD DATA SNOOPINGU KONFIGURACJA ---
HUB_QUBIT = 3
OPTIMAL_WEIGHTS = [0.10, 0.88, 0.26, 1.00, 0.98]

FEATURES_MAP = [
    'pbv_pko', 'zmiennosc',
    'pbv_peo', 'rentownosc_10y',
    'pbv_san', 'wig20',
    'pbv_ing', 'wibor_3m',
    'pbv_mbk', 'eurpln'
]

BATCH_SIZE = 50
SHOTS = 4000

RESULTS_CSV = sp("stability_qpu_odra_results_WALIDACJA.csv")
SUMMARY_CSV = sp("stability_qpu_odra_summary_WALIDACJA.csv")

# ==========================================
# 1b. WCZYTANIE GRANICY SELEKCJA/WALIDACJA
# ==========================================
split_config_path = sp("split_config.json")
if not os.path.exists(split_config_path):
    raise FileNotFoundError(
        f"Brak {split_config_path}. Ten test MUSI używać tej samej granicy "
        f"selekcja/walidacja co skrypt doboru huba/wag (Etap 1). "
        f"Skopiuj split_config.json do folderu tego skryptu przed uruchomieniem."
    )
with open(split_config_path) as f:
    SPLIT = json.load(f)

SPLIT_IDX = SPLIT.get("split_idx", SPLIT.get("split_t_index_in_df"))
if SPLIT_IDX is None:
    raise KeyError("split_config.json nie zawiera 'split_idx' ani 'split_t_index_in_df'.")

print(f">>> Wczytano split_config.json")
print(f">>> Granica selekcja/walidacja: indeks {SPLIT_IDX} (data: {SPLIT.get('split_date', '???')})")
print(f">>> TEN SKRYPT BĘDZIE OCENIAĆ WYŁĄCZNIE t >= {SPLIT_IDX} (okres walidacyjny)")

# ==========================================
# 2. PODŁĄCZENIE DO ODRY
# ==========================================
print("\n>>> Inicjalizacja połączenia z IQM Odra...")
load_dotenv(sp("token.env"))
server_url = os.getenv("SERVER")
token_val = os.getenv("TOKEN")

if not server_url:
    raise ValueError(f"Brak adresu SERVER w pliku {sp('token.env')}!")

provider = IQMProvider(server_url, token=token_val)
backend = provider.get_backend("Odra")
print(f">>> Sukces: Podłączono do backendu {backend.name}")

# ==========================================
# 3. PRZYGOTOWANIE BAZY DANYCH
# ==========================================
csv_path = sp("dataset_final.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Brak {csv_path}. Umieść dataset_final.csv w folderze skryptu.")

df = pd.read_csv(csv_path, index_col=0)
df.index = pd.to_datetime(df.index)
df = df.sort_index()

X_raw = df[FEATURES_MAP].values
y_raw = df["target_y"].astype(int).values
dates_raw = df.index.values

X_norm_list, valid_indices = [], []
for t in range(TRAIN_WINDOW, len(X_raw)):
    past_window = X_raw[t - TRAIN_WINDOW: t]
    min_vals = past_window.min(axis=0)
    max_vals = past_window.max(axis=0)
    range_vals = np.where((max_vals - min_vals) == 0, 1e-8, max_vals - min_vals)
    current_scaled = ((X_raw[t] - min_vals) / range_vals) * (2 * np.pi)
    X_norm_list.append(np.clip(current_scaled, 0.0, 2 * np.pi))
    valid_indices.append(t)

X_quantum_base = np.array(X_norm_list)
X_classical_valid = X_raw[valid_indices]
y = y_raw[valid_indices]
dates = dates_raw[valid_indices]

multipliers = np.zeros(10)
for q in range(5):
    multipliers[2 * q] = OPTIMAL_WEIGHTS[q]
    multipliers[2 * q + 1] = OPTIMAL_WEIGHTS[q]
X_quantum_opt = X_quantum_base * multipliers

# ==========================================
# 3b. KLUCZOWE: OGRANICZENIE DO OKRESU WALIDACYJNEGO
# ==========================================
validation_start_pos = next(
    (i for i, t in enumerate(valid_indices) if t >= SPLIT_IDX), None
)
if validation_start_pos is None:
    raise ValueError("Cały zakres danych jest przed granicą selekcji — sprawdź split_config.json.")

n_before_cut = len(y)
X_quantum_opt = X_quantum_opt[validation_start_pos:]
X_classical_valid = X_classical_valid[validation_start_pos:]
y = y[validation_start_pos:]
dates = dates[validation_start_pos:]

print(f"\n>>> PRZYCIĘTO DANE DO OKRESU WALIDACYJNEGO:")
print(f"    Przed przycięciem: {n_before_cut} tygodni")
print(f"    Po przycięciu:     {len(y)} tygodni (t >= {SPLIT_IDX})")

if len(y) <= TRAIN_WINDOW:
    raise ValueError(
        f"Okres walidacyjny ({len(y)} tygodni) jest krótszy niż TRAIN_WINDOW "
        f"({TRAIN_WINDOW}) — nie da się zbudować okna treningowego."
    )

# ==========================================
# 4. RETRY / ODPORNOŚĆ NA BŁĘDY SIECI
# ==========================================
MAX_RETRY_DELAY = 300  
INITIAL_RETRY_DELAY = 5  

def run_job_with_infinite_retry(physical_backend, batch, shots, context_label=""):
    attempt = 0
    delay = INITIAL_RETRY_DELAY
    while True:
        attempt += 1
        try:
            job = physical_backend.run(batch, shots=shots)
            result = job.result()  
            return result
        except KeyboardInterrupt:
            raise
        except Exception as e:
            err_text = f"{type(e).__name__}: {e}"
            print(f"\n[RETRY] Próba {attempt} nieudana ({context_label}). Błąd: {err_text}")
            print(f"[RETRY] Czekam {delay}s przed kolejną próbą...")
            time.sleep(delay)
            delay = min(delay * 2, MAX_RETRY_DELAY)
            if attempt % 5 == 0:
                try:
                    print("[RETRY] Odświeżanie połączenia z IQM Provider...")
                    fresh_provider = IQMProvider(server_url, token=token_val)
                    fresh_backend = fresh_provider.get_backend("Odra")
                    physical_backend = fresh_backend
                    print("[RETRY] Połączenie odświeżone.")
                except Exception as e2:
                    print(f"[RETRY] Odświeżenie nieudane: {e2}. Próbuję dalej ze starym backendem.")

def transpile_with_infinite_retry(circuits, backend, optimization_level=3):
    attempt = 0
    delay = INITIAL_RETRY_DELAY
    while True:
        attempt += 1
        try:
            return transpile(circuits, backend=backend, optimization_level=optimization_level)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"\n[RETRY] Transpilacja nieudana (próba {attempt}). Błąd: {e}")
            print(f"[RETRY] Czekam {delay}s przed kolejną próbą...")
            time.sleep(delay)
            delay = min(delay * 2, MAX_RETRY_DELAY)

# ==========================================
# 5. FIZYCZNE WYKONANIE NA QPU
# ==========================================
def build_base_ansatz(x, num_qubits=5, hub_qubit=HUB_QUBIT):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(x[2 * i], i)
        qc.rz(x[2 * i + 1], i)
    for i in range(num_qubits):
        if i != hub_qubit:
            qc.cx(hub_qubit, i)
    return qc

def get_phi_physical_qpu(run_idx, X_q, physical_backend):
    w_tag = "_".join(f"{w:.2f}" for w in OPTIMAL_WEIGHTS)
    filename = sp(f"qpu_phi_VALIDATION_run_{run_idx}_hub{HUB_QUBIT}_{w_tag}.csv")

    if os.path.exists(filename):
        print(f"\n[CACHE] Znaleziono {filename}. Wczytuję pomiary sprzętowe z dysku.")
        return pd.read_csv(filename, index_col=0).values

    print(f"\n[HARDWARE] Generowanie obwodów dla Iteracji {run_idx} (okres walidacyjny, {len(X_q)} tygodni)...")
    all_circuits = []
    for x in X_q:
        qc_base = build_base_ansatz(x, 5, HUB_QUBIT)
        for p in ['X', 'Y', 'Z']:
            for q in range(5):
                qc_measure = qc_base.copy()
                if p == 'X':
                    qc_measure.h(q)
                elif p == 'Y':
                    qc_measure.sdg(q)
                    qc_measure.h(q)
                qc_measure.measure_all()
                all_circuits.append(qc_measure)

    print(f"[HARDWARE] Transpilacja pod układ fizyczny Odra (może zająć chwilę)...")
    transpiled = transpile_with_infinite_retry(all_circuits, physical_backend, optimization_level=3)

    all_counts = []
    total_batches = (len(transpiled) - 1) // BATCH_SIZE + 1

    print(f"[HARDWARE] Podzielono na {total_batches} paczek. Rozpoczynam wysyłanie do API IQM.")
    for i in range(0, len(transpiled), BATCH_SIZE):
        batch = transpiled[i: i + BATCH_SIZE]
        current_batch = i // BATCH_SIZE + 1
        print(f"   -> [W KOLEJCE] Wysyłanie paczki {current_batch}/{total_batches}")

        result = run_job_with_infinite_retry(
            physical_backend, batch, SHOTS,
            context_label=f"run={run_idx}, paczka={current_batch}/{total_batches}"
        )

        for j in range(len(batch)):
            all_counts.append(result.get_counts(j))
        print(f"   -> [OK] Paczka {current_batch}/{total_batches} zakończona sukcesem.")

    print(f"[HARDWARE] Zakończono zrzuty kwantowe iteracji {run_idx}. Przeliczam...")

    Phi_qpu = np.zeros((len(X_q), 15))
    idx = 0
    for i in range(len(X_q)):
        for j in range(15):
            counts = all_counts[idx]
            total_shots = sum(counts.values())
            expval = 0
            for bitstring, count in counts.items():
                bits = [1 if b == '0' else -1 for b in reversed(bitstring.replace(" ", ""))]
                q_target = j % 5
                expval += bits[q_target] * count
            Phi_qpu[i, j] = expval / total_shots
            idx += 1

    pd.DataFrame(Phi_qpu).to_csv(filename)
    print(f"[HARDWARE] Zapisano stan macierzy do {filename}.")
    return Phi_qpu

# ==========================================
# 6. FUNKCJE POMOCNICZE (PRAWIDŁOWY THRESHOLD)
# ==========================================
def get_best_preds(probs, y_true):
    best_thresh, best_acc = 0.50, 0
    for thresh in np.arange(0.40, 0.60, 0.01):
        acc = accuracy_score(y_true, (probs > thresh).astype(int))
        if acc > best_acc:
            best_acc, best_thresh = acc, thresh
    return best_thresh

def get_threshold_from_training_window(pipeline_fitted, X_tr, y_tr):
    """Zwraca optymalny próg policzony WYŁĄCZNIE na oknie treningowym"""
    probs_tr = pipeline_fitted.predict_proba(X_tr)[:, 1]
    return get_best_preds(probs_tr, y_tr)

def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

# ==========================================
# 7. GŁÓWNA PĘTLA TESTU STABILNOŚCI QPU (TYLKO WALIDACJA)
# ==========================================
print(f"\n>>> Rozpoczynam Test Stabilności Hardware: {NUM_RUNS} iteracji Walk-Forward")
print(f">>> WYŁĄCZNIE na okresie walidacyjnym ({len(y)} tygodni, t >= {SPLIT_IDX})")
start_time_global = time.time()

pipeline = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", probability=True))])
inner_cv = TimeSeriesSplit(n_splits=3)
param_grid = {"svm__C": [0.1, 1.0, 5.0, 10.0], "svm__gamma": ["scale", "auto"]}

all_results = []
net_hits_history = {"Quantum": [], "Classical": []}

for run in range(1, NUM_RUNS + 1):

    Phi_real = get_phi_physical_qpu(run, X_quantum_opt, backend)

    results = {"y_true": [], "prob_qpu": [], "prob_cls": [], "pred_qpu": [], "pred_cls": [], "th_qpu": [], "th_cls": []}
    total_steps = len(y) - TRAIN_WINDOW

    print(f"\n[ML] Trening maszyn SVM dla iteracji {run} na pobranych pomiarach "
          f"({total_steps} kroków walidacyjnych)...")
    
    for step, t in enumerate(range(TRAIN_WINDOW, len(y))):
        Phi_tr, Phi_te = Phi_real[t - TRAIN_WINDOW: t], Phi_real[t: t + 1]
        X_cls_tr, X_cls_te = X_classical_valid[t - TRAIN_WINDOW: t], X_classical_valid[t: t + 1]
        y_tr = y[t - TRAIN_WINDOW: t]

        results["y_true"].append(y[t])
        pipeline.set_params(svm__random_state=(42 * run + t))

        # --- QUANTUM SVM ---
        search_qpu = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring="roc_auc", n_jobs=-1).fit(Phi_tr, y_tr)
        p_qpu = search_qpu.predict_proba(Phi_te)[0, 1]
        
        # Wyliczenie progu TYLKO na Phi_tr i y_tr (dane z przeszłości)
        th_qpu = get_threshold_from_training_window(search_qpu, Phi_tr, y_tr)
        
        results["prob_qpu"].append(p_qpu)
        results["pred_qpu"].append(1 if p_qpu > th_qpu else 0)
        results["th_qpu"].append(th_qpu)

        # --- CLASSICAL SVM ---
        search_cls = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring="roc_auc", n_jobs=-1).fit(X_cls_tr, y_tr)
        p_cls = search_cls.predict_proba(X_cls_te)[0, 1]
        
        # Wyliczenie progu TYLKO na X_cls_tr i y_tr (dane z przeszłości)
        th_cls = get_threshold_from_training_window(search_cls, X_cls_tr, y_tr)
        
        results["prob_cls"].append(p_cls)
        results["pred_cls"].append(1 if p_cls > th_cls else 0)
        results["th_cls"].append(th_cls)

        if step % 10 == 0 or step == total_steps - 1:
            percent_done = int((step + 1) / total_steps * 100)
            bar = "=" * (percent_done // 5) + "-" * (20 - percent_done // 5)
            print(f"\r[{bar}] ML Step {step + 1}/{total_steps}", end="", flush=True)

    y_true_arr = np.array(results["y_true"])
    prob_qpu = np.array(results["prob_qpu"])
    prob_cls = np.array(results["prob_cls"])
    pred_qpu = np.array(results["pred_qpu"])
    pred_cls = np.array(results["pred_cls"])

    net_hits_history["Quantum"].append(np.cumsum(np.where(pred_qpu == y_true_arr, 1, -1)))
    net_hits_history["Classical"].append(np.cumsum(np.where(pred_cls == y_true_arr, 1, -1)))

    per_week_path = sp(f"predictions_per_week_VALIDATION_run_{run}.csv")
    pd.DataFrame({
        "t_local": list(range(TRAIN_WINDOW, len(y))),
        "date": dates[TRAIN_WINDOW:len(y)],
        "y_true": y_true_arr,
        "prob_qpu": prob_qpu,
        "pred_qpu": pred_qpu,
        "prob_cls": prob_cls,
        "pred_cls": pred_cls,
        "th_qpu_used": results["th_qpu"],
        "th_cls_used": results["th_cls"]
    }).to_csv(per_week_path, index=False)

    for model_name, p, pred, th_list in [("Quantum SVM (Odra HW)", prob_qpu, pred_qpu, results["th_qpu"]),
                                         ("Classical SVM (Raw)", prob_cls, pred_cls, results["th_cls"])]:
        all_results.append({
            "Run": run,
            "Model": model_name,
            "N_walidacja": len(y_true_arr),
            "Accuracy": accuracy_score(y_true_arr, pred),
            "AUC ROC": roc_auc_score(y_true_arr, p),
            "F1 Score": f1_score(y_true_arr, pred),
            "MCC": matthews_corrcoef(y_true_arr, pred),
            "Opt Threshold (Mean)": np.mean(th_list)  # Zapisujemy średni próg użyty z całego Walk-Forward
        })

    pd.DataFrame(all_results).to_csv(RESULTS_CSV, index=False)
    print(f"\n[INFO] Zakończono analizę iteracji {run}. Zapisano wyniki do {RESULTS_CSV}")

total_exec_time = time.time() - start_time_global
print(f"\n>>> Zakończono cały test statystyczny! Całkowity czas: {format_time(total_exec_time)}")

# ==========================================
# 8. GENEROWANIE WIZUALIZACJI I WYNIKÓW
# ==========================================
df_res = pd.DataFrame(all_results)
summary_stats = df_res.groupby("Model").agg({
    "Accuracy": ['mean', 'std'],
    "AUC ROC": ['mean', 'std'],
    "F1 Score": ['mean', 'std'],
    "MCC": ['mean', 'std'],
    "Opt Threshold (Mean)": ['mean']
})
summary_stats.columns = [f"{col[0]}_{col[1]}" for col in summary_stats.columns]
summary_stats.to_csv(SUMMARY_CSV)

print("\n--- PODSUMOWANIE HARDWARE STABILITY (TYLKO OKRES WALIDACYJNY) ---")
print(summary_stats)

metrics_to_plot = ["Accuracy", "AUC ROC", "F1 Score", "MCC"]
colors = {"Quantum SVM (Odra HW)": "#8e44ad", "Classical SVM (Raw)": "#27ae60"}

for metric in metrics_to_plot:
    plt.figure(figsize=(10, 6))
    for model in colors.keys():
        model_data = df_res[df_res["Model"] == model]
        plt.plot(model_data["Run"], model_data[metric], marker='o', lw=2.5,
                 color=colors[model], label=f'{model} (Mean: {model_data[metric].mean():.3f} ± {model_data[metric].std():.3f})')

    plt.title(f'{metric} Stability on Physical QPU — VALIDATION ONLY ({NUM_RUNS} Iterations)',
              fontweight='bold', fontsize=14)
    plt.xlabel('Run Number', fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.xticks(range(1, NUM_RUNS + 1))
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(sp(f"Hardware_Stability_VALIDATION_{metric.replace(' ', '_')}.png"), dpi=300, bbox_inches='tight')
    plt.close()

plt.figure(figsize=(12, 7))
plot_dates = dates[TRAIN_WINDOW:len(y)]
for i in range(NUM_RUNS):
    plt.plot(plot_dates, net_hits_history["Quantum"][i], color="#8e44ad", alpha=0.3, lw=1.5)
    plt.plot(plot_dates, net_hits_history["Classical"][i], color="#27ae60", alpha=0.3, lw=1.5)

plt.plot(plot_dates, np.mean(net_hits_history["Quantum"], axis=0), color="#8e44ad", lw=3, label="Quantum SVM (Avg on QPU)")
plt.plot(plot_dates, np.mean(net_hits_history["Classical"], axis=0), color="#27ae60", lw=3, ls="--", label="Classical SVM (Avg)")
plt.axhline(0, color='black', lw=1.5)
plt.title('Cumulative Net Hits Stability — VALIDATION ONLY (QPU Odra)', fontweight='bold', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Net Advantage', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(sp("Hardware_Stability_VALIDATION_Cumulative_Net_Hits.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"\n>>> Wszystkie pliki wynikowe zapisano w: {SCRIPT_DIR}")