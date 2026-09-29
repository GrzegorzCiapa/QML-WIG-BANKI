"""
Analiza statystyczna eksperymentu Walk-Forward – sparowane porównanie AUC ROC:
    Quantum SVM (Odra HW)  vs  Classical SVM (Raw)

Uruchomienie:
    python analiza_walk_forward.py               # plik z CSV_PATH
    python analiza_walk_forward.py wyniki.csv    # dowolny inny plik CSV

Wymagania: pandas, numpy, scipy >= 1.7
"""

from __future__ import annotations

import platform
import re
import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats

# ================================ KONFIGURACJA ================================
# Bezwzględna ścieżka chroniąca przed błędem FileNotFoundError
CSV_PATH = "/home/gciapa/Projekt_QML_WIG/Projekt/big-bank_9)Finalny kod i testy stat/stability_qpu_odra_results_WALIDACJA.csv"

COL_RUN, COL_MODEL, COL_AUC = "Run", "Model", "AUC ROC"
MODEL_QUANTUM = "Quantum SVM (Odra HW)"
MODEL_CLASSICAL = "Classical SVM (Raw)"

PLANNED_RUNS = 10          # planowana liczba iteracji (do komunikatu o wynikach wstępnych)
ALPHA = 0.05               # poziom istotności (używany w uwagach)
CONFIDENCE = 0.95          # poziom ufności przedziałów CI
N_BOOTSTRAP = 10_000       # liczba resampli bootstrap
BOOTSTRAP_METHOD = "BCa"   # "BCa" (domyślna w SciPy) | "percentile" | "basic"
SEED = 42                  # ziarno RNG -> powtarzalny przedział bootstrap
DECIMALS = 4               # miejsca po przecinku w raporcie
# ==============================================================================


class DataError(Exception):
    """Problem z plikiem wejściowym (format, brakujące kolumny lub modele)."""


# ------------------------------- Wczytanie danych -------------------------------

def _normalize(text) -> str:
    return re.sub(r"[^0-9a-z]", "", str(text).lower())


def _find_column(df: pd.DataFrame, wanted: str) -> str:
    for col in df.columns:
        if _normalize(col) == _normalize(wanted):
            return col
    raise DataError(f"brak kolumny '{wanted}'. Kolumny w pliku: {list(df.columns)}")


def _find_model(models: list[str], wanted: str) -> str:
    if wanted in models:
        return wanted
    for model in models:
        if _normalize(model) == _normalize(wanted):
            return model
    raise DataError(f"brak modelu '{wanted}' w kolumnie '{COL_MODEL}'. Dostępne modele: {models}")


def load_data(path: Path) -> pd.DataFrame:
    """Wczytuje CSV (separator , ; lub TAB wykrywany automatycznie) i zwraca kolumny Run, Model, AUC ROC."""
    try:
        try:
            raw = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
        except UnicodeDecodeError:
            raw = pd.read_csv(path, sep=None, engine="python", encoding="cp1250")
    except Exception as exc:
        raise DataError(f"nie udało się wczytać pliku CSV ({exc})") from exc

    df = pd.DataFrame({col: raw[_find_column(raw, col)] for col in (COL_RUN, COL_MODEL, COL_AUC)})
    df[COL_MODEL] = df[COL_MODEL].astype(str).str.strip()
    df[COL_RUN] = pd.to_numeric(df[COL_RUN], errors="coerce")
    if not pd.api.types.is_numeric_dtype(df[COL_AUC]):  # np. przecinek dziesiętny (eksport z Excela)
        df[COL_AUC] = df[COL_AUC].astype(str).str.strip().str.replace(",", ".", regex=False)
    df[COL_AUC] = pd.to_numeric(df[COL_AUC], errors="coerce")
    return df


def build_pairs(df: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    """Tabela sparowana po Run (tylko kompletne pary): quantum, classical, diff = quantum − classical."""
    if df.empty:
        raise DataError("plik nie zawiera jeszcze żadnych wyników")
    models = sorted(df[COL_MODEL].dropna().unique())
    model_q = _find_model(models, MODEL_QUANTUM)
    model_c = _find_model(models, MODEL_CLASSICAL)

    sub = df[df[COL_MODEL].isin([model_q, model_c])]
    invalid = sub[COL_RUN].isna() | sub[COL_AUC].isna()
    if invalid.any():
        notes.append(f"Pominięte wiersze z brakującą lub nienumeryczną wartością "
                     f"'{COL_RUN}'/'{COL_AUC}': {int(invalid.sum())}.")
    sub = sub[~invalid].copy()
    if (sub[COL_RUN] % 1 == 0).all():
        sub[COL_RUN] = sub[COL_RUN].astype(int)

    duplicated = sub.duplicated(subset=[COL_RUN, COL_MODEL], keep="last")
    if duplicated.any():
        runs = ", ".join(str(run) for run in sorted(sub.loc[duplicated, COL_RUN].unique()))
        notes.append(f"Zduplikowane wpisy (Run, Model) dla Run: {runs} – użyto ostatniego wpisu z pliku.")
        sub = sub[~duplicated]

    wide = (sub.pivot(index=COL_RUN, columns=COL_MODEL, values=COL_AUC)
                .reindex(columns=[model_q, model_c])
                .sort_index())
    for run, row in wide[wide.isna().any(axis=1)].iterrows():
        missing = " i ".join(f"'{m}'" for m in (model_q, model_c) if pd.isna(row[m]))
        notes.append(f"Run {run} pominięty – brak wyniku dla {missing}.")

    complete = wide.dropna()
    paired = pd.DataFrame({"quantum": complete[model_q], "classical": complete[model_c]})
    # Zaokrąglenie do 12 miejsc usuwa szum zmiennoprzecinkowy (rzędu 1e-17),
    # dzięki czemu zera i remisy |Δ| w teście Wilcoxona są wykrywane poprawnie.
    paired["diff"] = (paired["quantum"] - paired["classical"]).round(12)
    return paired


# ---------------------------------- Statystyki ----------------------------------

def _safe_call(name: str, notes: list[str], func, *args, **kwargs):
    """Wywołuje funkcję SciPy; ostrzeżenia i wyjątki trafiają do sekcji 'Uwagi' zamiast przerywać raport."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = func(*args, **kwargs)
        except Exception as exc:
            notes.append(f"{name}: {exc}")
            result = None
    for w in caught:
        if not issubclass(w.category, (DeprecationWarning, PendingDeprecationWarning, FutureWarning)):
            notes.append(f"{name}: {w.message}")
    return result


def _bootstrap_mean_ci(d: np.ndarray) -> tuple[float, float]:
    options = dict(n_resamples=N_BOOTSTRAP, confidence_level=CONFIDENCE, method=BOOTSTRAP_METHOD)
    try:
        res = stats.bootstrap((d,), np.mean, rng=np.random.default_rng(SEED), **options)
    except TypeError:  # starsze wersje SciPy przyjmują tylko `random_state`
        res = stats.bootstrap((d,), np.mean, random_state=np.random.default_rng(SEED), **options)
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def compute_statistics(paired: pd.DataFrame, notes: list[str]) -> dict:
    """Statystyki wektora różnic. Wartość typu str oznacza powód, dla którego statystyki nie policzono."""
    q = paired["quantum"].to_numpy(dtype=float)
    c = paired["classical"].to_numpy(dtype=float)
    d = paired["diff"].to_numpy(dtype=float)
    n = d.size
    sd = float(np.std(d, ddof=1)) if n >= 2 else np.nan

    if n < 2:
        blocker = "wymaga n ≥ 2"
    elif d.max() == d.min():
        blocker = "SD różnic = 0"
    else:
        blocker = None

    r = {
        "n": n,
        "n_pos": int(np.sum(d > 0)),
        "mean": float(np.mean(d)),
        "sd": sd if n >= 2 else "wymaga n ≥ 2",
    }

    # Shapiro-Wilk: normalność różnic
    if n < 3:
        r["shapiro"] = "wymaga n ≥ 3"
    elif blocker:
        r["shapiro"] = blocker
    else:
        res = _safe_call("Test Shapiro-Wilka", notes, stats.shapiro, d)
        r["shapiro"] = tuple(map(float, res)) if res is not None else "patrz Uwagi"

    # Wilcoxon signed-rank, H1: Δ > 0 (zerowe różnice odrzucane, zero_method='wilcox')
    n_nonzero = int(np.count_nonzero(d))
    if n_nonzero == 0:
        r["wilcoxon_p"] = "wszystkie różnice = 0"
    else:
        if n_nonzero < n:
            notes.append(f"Test Wilcoxona: pominięto zerowe różnice ({n - n_nonzero}), "
                         f"efektywne n = {n_nonzero}.")
        min_p = 0.5 ** n_nonzero
        if min_p > ALPHA:
            notes.append(f"Test Wilcoxona: przy n = {n_nonzero} najmniejsze osiągalne p (test dokładny, "
                         f"jednostronny) wynosi 1/2^{n_nonzero} = {_num(min_p)} > α = {ALPHA}, "
                         f"więc istotności nie da się uzyskać niezależnie od wyników.")
        res = _safe_call("Test Wilcoxona", notes, stats.wilcoxon, d, alternative="greater")
        r["wilcoxon_p"] = float(res.pvalue) if res is not None else "patrz Uwagi"

    # Statystyki wymagające n ≥ 2 i niezerowej wariancji różnic
    if blocker:
        r.update(dict.fromkeys(("ttest", "ci_boot", "ci_t", "cohen_d"), blocker))
        return r

    res = _safe_call("Test t-Studenta", notes, stats.ttest_rel, q, c, alternative="greater")
    r["ttest"] = (float(res.statistic), float(res.pvalue)) if res is not None else "patrz Uwagi"

    ci = _safe_call("Bootstrap", notes, _bootstrap_mean_ci, d)
    r["ci_boot"] = ci if ci is not None and np.all(np.isfinite(ci)) else "patrz Uwagi"

    lo, hi = stats.t.interval(CONFIDENCE, n - 1, loc=r["mean"], scale=stats.sem(d))
    r["ci_t"] = (float(lo), float(hi))

    r["cohen_d"] = r["mean"] / sd
    return r


# ------------------------------------ Raport ------------------------------------

def _num(x: float, signed: bool = False) -> str:
    if not np.isfinite(x):
        return "n/d"
    x = round(float(x), DECIMALS) + 0.0  # "+ 0.0" zamienia -0.0 na 0.0 (bez "-0.0000")
    return f"{x:+.{DECIMALS}f}" if signed else f"{x:.{DECIMALS}f}"


def _pval(p: float) -> str:
    limit = 10.0 ** -DECIMALS
    return f"p < {limit:.{DECIMALS}f}" if p < limit else f"p = {_num(p)}"


def _ci(bounds: tuple[float, float]) -> str:
    return f"[{_num(bounds[0])}, {_num(bounds[1])}]"


def _cell(value, render) -> str:
    return f"n/d ({value})" if isinstance(value, str) else render(value)


def print_report(csv_path: Path, paired: pd.DataFrame, r: dict, notes: list[str]) -> None:
    n = r["n"]
    ci_name = f"{CONFIDENCE:.0%} CI dla Δ"
    n_boot = f"{N_BOOTSTRAP:,}".replace(",", " ")
    rows = [
        ("Liczba par (iteracji)", str(n)),
        ("Liczba iteracji z dodatnią różnicą (Quantum > Classical)", f"{r['n_pos']}/{n}"),
        ("Średnia różnica AUC (Δ)", _num(r["mean"])),
        ("Odchylenie standardowe różnic", _cell(r["sd"], _num)),
        ("Test Shapiro-Wilka (normalność różnic)",
         _cell(r["shapiro"], lambda v: f"W = {_num(v[0])}, {_pval(v[1])}")),
        ("Test Wilcoxona (signed-rank, jednostronny)", _cell(r["wilcoxon_p"], _pval)),
        ("Sparowany test t-Studenta (jednostronny)",
         _cell(r["ttest"], lambda v: f"t({n - 1}) = {_num(v[0])}, {_pval(v[1])}")),
        (f"{ci_name} (bootstrap, {n_boot} resampli)", _cell(r["ci_boot"], _ci)),
        (f"{ci_name} (analityczny, rozkład t)", _cell(r["ci_t"], _ci)),
        ("Cohen's d (obserwowane)", _cell(r["cohen_d"], _num)),
    ]
    label_w = max(len(label) for label, _ in rows) + 3
    width = 2 + label_w + max(28, max(len(value) for _, value in rows))
    heavy, light = "═" * width, "─" * width

    def per_run_row(first: str, quantum: str, classical: str, diff: str) -> None:
        print(f"  {first:>7}{quantum:>16}{classical:>17}{diff:>15}")

    print(f"\n{heavy}")
    print("  RAPORT STATYSTYCZNY – EKSPERYMENT WALK-FORWARD (próby sparowane, AUC ROC)")
    print(heavy)
    print(f"  Plik danych : {csv_path.name}")
    print(f"  Modele      : {MODEL_QUANTUM}  vs  {MODEL_CLASSICAL}  (pary wg '{COL_RUN}')")
    print("  Różnica     : Δ = AUC(Quantum) − AUC(Classical)")
    print("  Hipotezy    : H0: Δ ≤ 0  vs  H1: Δ > 0  (testy jednostronne)")

    print(f"\n  Tabela 1. AUC ROC w poszczególnych iteracjach (n = {n})")
    print(light)
    per_run_row("Run", "AUC Quantum", "AUC Classical", "Δ (Q − C)")
    print(light)
    for run, row in paired.iterrows():
        per_run_row(str(run), _num(row["quantum"]), _num(row["classical"]), _num(row["diff"], signed=True))
    print(light)
    mean = paired.mean()
    per_run_row("Średnia", _num(mean["quantum"]), _num(mean["classical"]), _num(mean["diff"], signed=True))
    if n >= 2:
        sd = paired.std(ddof=1)
        per_run_row("SD", _num(sd["quantum"]), _num(sd["classical"]), _num(sd["diff"]))

    print("\n  Tabela 2. Statystyki opisowe i testy hipotez dla różnic Δ")
    print(light)
    print(f"  {'Statystyka':<{label_w}}Wartość")
    print(light)
    for label, value in rows:
        print(f"  {label:<{label_w}}{value}")
    print(light)
    print(f"  SD z ddof = 1. Bootstrap: metoda {BOOTSTRAP_METHOD}, seed = {SEED}. Cohen's d = Δ / SD (d_z).")
    print(f"  Środowisko: Python {platform.python_version()}, pandas {pd.__version__}, "
          f"NumPy {np.__version__}, SciPy {scipy.__version__}")

    if notes:
        print("\n  Uwagi:")
        for note in dict.fromkeys(notes):
            print(textwrap.fill(note, width=width - 2, initial_indent="   • ", subsequent_indent="     "))
    print(heavy)


# ------------------------------------- Main -------------------------------------

def _enable_utf8_output() -> None:
    # Przy przekierowaniu wyjścia Windows używa cp1250, w którym brak znaków Δ, ≤, −.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError, OSError):
            pass


def resolve_csv_path() -> Path:
    """Plik z argumentu wywołania (*.csv) lub CSV_PATH; ścieżka względna szukana też obok skryptu."""
    cli_arg = next((arg for arg in sys.argv[1:] if arg.lower().endswith(".csv")), None)
    path = Path(cli_arg or CSV_PATH)
    if not path.is_file() and not path.is_absolute() and "__file__" in globals():
        candidate = Path(__file__).resolve().parent / path
        if candidate.is_file():
            return candidate
    return path


def main() -> None:
    _enable_utf8_output()
    csv_path = resolve_csv_path()
    if not csv_path.is_file():
        sys.exit(f"BŁĄD: nie znaleziono pliku '{csv_path}'.")

    notes: list[str] = []
    try:
        paired = build_pairs(load_data(csv_path), notes)
    except DataError as exc:
        sys.exit(f"BŁĄD: {exc}")

    if paired.empty:
        sys.exit("BŁĄD: brak kompletnych par (Run z wynikami obu modeli).\n"
                 + "\n".join(f"  • {note}" for note in notes))
    if len(paired) < PLANNED_RUNS:
        notes.insert(0, f"Wyniki wstępne: dostępne {len(paired)} z {PLANNED_RUNS} planowanych iteracji.")

    results = compute_statistics(paired, notes)
    print_report(csv_path, paired, results, notes)


if __name__ == "__main__":
    main()