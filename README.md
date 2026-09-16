# QML-WIG20

### Co jest zrobione:
* Wstępnie postawiłem model kwantowy, który normalnie się uczy i nawet sensownie ewaluuje w pętli kroczącej (cokolwiek to znaczy XD).
* Wrzuciłem kod na Odrę. 
* Wyniki są w pliku `dane/phi_odra_wig_banki.npy`. Nie ma sensu znowu tego wrzucać, liczyło się długo i robiłbym to tylko jak zmienimy strukturę modelu.
* Na gotowym pliku z rzutami można od ręki odpalać klasyfikator i bawić się parametrami.

### Wyniki na ten moment:
* Średnia korelacja rzutów Odra vs idealny symulator: `0.9200` O dziwo zajebisty wynik.
* Directional Accuracy (QPU Odra):** `52.58%` (dla porównania: klasyczna regresja logistyczna wypluła `50.80%`, analityczny symulator `50.98%`). Naturalny szum Odry zadziałał trochę jak regularyzacja(?) i podbił wynik względem idealnej symulacji ale nie wiem czy to dobrze.

### Co jest dalej do zrobienia:
* Trzeba pobawić się parametrami modelu, może uda się dostać lepsze wyniki. ()
* Czat wygenerował mi jakieś wykresy, ale w sumie to nie wiem co na nich jest XD. Do tego trzeba dodać jakieś krzywe treningowe i wykresy wizualizacyjne, żeby wyglądało profesjonalnie i było wiadomo o co chodzi. Jakieś porównanie ect...

---
# Algorytm PQK-SVM - wersja 16.09.2026:
## Opis Projektu

Projekt bada zastosowanie 5-kubitowego komputera kwantowego (IQM Odra/Spark) do przewidywania kierunku zmian giełdowego indeksu WIG-Banki w horyzoncie jednego tygodnia. Głównym celem jest weryfikacja, czy nieliniowe przekształcenia w przestrzeni kwantowej potrafią skuteczniej wychwycić rynkowe zależności niż standardowe algorytmy klasyczne (np. Support Vector Machine, Regresja Logistyczna).

Eksperyment obejmuje bezpośrednie porównanie klasycznych metod uczenia maszynowego z algorytmem kwantowym, uruchamianym zarówno w środowisku idealnego symulatora, jak i na fizycznym, zaszumionym sprzęcie QPU.

## Architektura i Kodowanie Danych

Projekt wykorzystuje metodę **Projected Quantum Kernel (PQK)**. 10 historycznych cech wejściowych (znormalizowanych wskaźników fundamentalnych i makroekonomicznych) zostało skompresowanych na 5 fizycznych kubitach za pomocą techniki **Dense Angle Encoding**.

Stan każdego kubitu jest modyfikowany według wzoru:


$$\vert{}\psi_i\rangle = R_z(\pi \cdot x_{makro}) R_y(\pi \cdot x_{bank}) \vert{}0\rangle$$

### Mapowanie Cech na Kubity:

| Kubit | Wycena Banku (Bramka $R_y$) | Czynnik Rynkowy (Bramka $R_z$) | Uzasadnienie |
| --- | --- | --- | --- |
| **0** | P/BV PKO BP | Stawka WIBOR 3M | Korelacja marży odsetkowej największego banku ze stopami proc. |
| **1** | P/BV Pekao | Rentowność obligacji 10y | Wrażliwość dużego portfela papierów dłużnych na krzywą rentowności. |
| **2** | P/BV Santander BP | Kurs EUR/PLN | Wpływ siły polskiego złotego na ocenę banku z globalnej grupy kapitałowej. |
| **3** | P/BV ING BSK | Indeks WIG20 | Ścisły związek stabilnej wyceny komercyjnej z szerokim sentymentem giełdy. |
| **4** | P/BV mBank | Zmienność (Z-Score) | Ekspozycja na ryzyko prawne (kredyty CHF) warunkująca najwyższą zmienność. |

Zamiast kosztownego szacowania pełnego stanu kwantowego, model wylicza lokalne wartości oczekiwane obserwabli (X, Y, Z) na każdym z 5 kubitów, tworząc 15-wymiarowy wektor klasyczny, który następnie trafia do klasycznego klasyfikatora SVM z jądrem RBF.

## Metodyka Ewaluacji

* **Horyzont Predykcji:** 1 tydzień giełdowy.
* **Walidacja:** Walk-Forward (Out-Of-Sample) z przesuwnym oknem treningowym wynoszącym 52 tygodnie (1 rok). Chroni to model przed "wyciekiem danych z przyszłości" (data leakage).
* **Główne Metryki:** AUC-ROC (zdolność rozróżniania klas), F1-Score oraz Skumulowany Bilans Trafień (Equity Curve).

## Wymagania Środowiskowe

Aby uruchomić kod, wymagane jest środowisko Python 3.9+ oraz zainstalowane biblioteki:

* `qiskit`, `qiskit-iqm`, `qiskit-aer`
* `scikit-learn`, `pandas`, `numpy`, `matplotlib`
* `python-dotenv`

## Struktura Plików

* `main.py` / `notebook.ipynb` – Główny skrypt eksperymentu badawczego.
* `dataset_final.csv` – Zagregowane i znormalizowane dane tygodniowe dla banków i wskaźników makro.
* `iqm_token.env` – Plik konfiguracyjny (zmienne `IQM_TOKEN` oraz `SERVER`) służący do autoryzacji z procesorem IQM Odra.
* `phi_sim_ideal.npy` – Zapisane w pamięci podręcznej (cache) rzuty wektorów z symulatora bezszumowego.
* `phi_real_odra.npy` – Wyniki rzutowania pobrane z fizycznego komputera kwantowego (tworzone/nadpisywane podczas strzałów na QPU).

## Uruchomienie Eksperymentu

1. Upewnij się, że plik `dataset_final.csv` oraz `iqm_token.env` znajdują się w tym samym katalogu (lub podkatalogu `dane/`).
2. Uruchom skrypt. Przy pierwszym uruchomieniu algorytm wymusi połączenie z fizycznym procesorem IQM, przetranspiluje obwody i wyśle zadania (batch) po `4000 shots` każde.
3. Zrzut z komputera kwantowego zostanie zapisany na dysku. Każda kolejna ewaluacja w pętli Walk-Forward wczyta go w ułamek sekundy do optymalizacji hiperparametrów klasyfikatora.
4. Na zakończenie wygenerowany zostanie graficzny **Dashboard Ostateczny** przedstawiający krzywe ROC i wierność odwzorowania rynku.

## Co do zrobienia?:

### 1. Dynamiczny Tuning Hiperparametrów

Obecnie klasyfikatory (klasyczny SVM i PQK) mają wpisane `C=5.0` na sztywno. Rynek finansowy jest bardzo zmienny, więc optymalny margines błędu klasyfikatora powinien się adaptować do bieżącego reżimu.

* Zmodyfikuj pętlę Walk-Forward, aby przed wygenerowaniem predykcji (na oknie treningowym) uruchamiała wewnętrzną, zagnieżdżoną walidację krzyżową (np. `GridSearchCV`).
* Niech algorytm sam w locie testuje siatkę parametrów `C` (np. 0.1, 1.0, 5.0, 10.0) i używa tego, który w danym roku giełdowym działał najlepiej.

### 2. Architektura Obwodu Kwantowego (Ansatz Engineering)

Twoja funkcja `build_base_ansatz` tworzy dokładnie jedną warstwę rotacji i jedną warstwę splątania (bramki CNOT).

* Dodaj do funkcji parametr określający liczbę powtórzeń (głębokość obwodu).
* Przeprowadź i zapisz eksperyment testujący obwód z 2 oraz 3 warstwami.
* Sprawdź, w którym momencie głębszy obwód poprawia nieliniowość i wynik AUC, a kiedy generuje zbyt duży szum sprzętowy, który degraduje sygnał z Odry.

### 3. Optymalizacja Progu Decyzyjnego

Linijka z przypisaniem klasy zakłada symetryczny próg wejścia w pozycję wynoszący równo 50%. W zaawansowanych strategiach finansowych rzadko to się sprawdza.

* Napisz funkcję, która wewnątrz okna treningowego szuka takiego progu prawdopodobieństwa (np. 0.54 dla wzrostów i 0.46 dla spadków), który maksymalizuje wskaźnik F1-Score.
* Zastosuj ten dynamicznie wyuczony próg na danych testowych. Zmusi to algorytm do zajmowania pozycji tylko w momentach silnej pewności.

### 4. Analiza Ważności Cech (Ablation Study)

Recenzenci publikacji na pewno zapytają, który z pięciu banków i wskaźników makroekonomicznych ułatwił algorytmowi zadanie w największym stopniu.

* Przygotuj skrypt "uszkadzający" obwód, który uruchamia ewaluację PQK pięć razy, za każdym razem wyłączając jeden z fizycznych kubitów z procesu pomiarowego.
* Zmierz i zapisz spadek metryki AUC-ROC dla każdego usuniętego kubitu. Ten czynnik makro, którego wyłączenie najbardziej obniża skuteczność modelu, jest jego najważniejszym silnikiem.
