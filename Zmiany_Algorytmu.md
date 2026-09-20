## Etap 1: Inżynieria Danych i Pierwszy Prototyp (`big_bank_1)`)



* Skrypt ładuje historyczne wyceny P/BV pięciu polskich banków, z których buduje syntetyczny benchmark sektorowy WIG-Banki (baza 1000 pkt).
* Obliczana jest 20-dniowa krocząca zmienność indeksu WIG20 (Z-score) oraz wczytywane są dane makroekonomiczne (WIBOR, EURPLN, rentowność obligacji).
* Zbudowano pierwszą wersję 5-kubitowego obwodu w oparciu o technikę Dense Angle Encoding (1 warstwa rotacji $R_y$, $R_z$ i 1 warstwa splątania).
* Skrypt uruchamia ewaluację Out-Of-Sample, zestawiając klasyczną regresję logistyczną z modelem Projected Quantum Kernel (PQK) działającym na idealnym symulatorze oraz fizycznym procesorze IQM Odra, zapisując zrzut sprzętowy do pliku `.npy`.

## Etap 2: Rygor Walidacyjny i Uszczelnienie Potoku (`big_bank_2)Hiperparametry`)

 — **Ślepa droga**

Architektura izolacji danych (Pipeline): Do potoku wprowadzono obiekty Pipeline, co zagwarantowało, że transformacje cech zachodziły ściśle wewnątrz bieżącego okna treningowego (dla modeli klasycznych użyto MinMaxScaler(), a dla projekcji kwantowych StandardScaler()), eliminując błąd wycieku informacji (data leakage).
Wielowymiarowy tuning kroczący (RandomizedSearchCV + TimeSeriesSplit): W każdym kroku procedury Walk-Forward hiperparametry były dobierane dynamicznie z użyciem 5-krotnego podziału szeregu czasowego. Siatka obejmowała regularyzację C, parametr skali γ (dla jądra RBF) oraz wagowanie klas (class_weight), optymalizowane bezpośrednio pod kątem maksymalizacji wskaźnika ROC AUC.
Rozszerzenie horyzontu treningowego (104 tygodnie): Okno bazowe wydłużono z 52 do 104 tygodni (2 lata), aby algorytmom optymalizacyjnym dostarczyć wystarczającą próbę do podziałów walidacji krzyżowej bez drastycznego ubytku danych uczących.
Wnioski badawcze („Ślepa droga” i niestacjonarność rynku): Rygorystyczny tuning obnażył niestacjonarny charakter rynku finansowego — optymalne parametry dopasowywały się do przeszłego reżimu, prowadząc do zjawiska ujemnego transferu i spadku AUC poniżej progu losowego (0.46–0.49). Mimo ogólnego pogorszenia metryk, fizyczne QPU utrzymało najwyższy wynik separacji w zestawieniu (AUC=0.4891) oraz dodatni bilans trafień w analizie Equity Curve, podczas gdy modele klasyczne zaliczyły głębokie obsunięcia.
## Etap 3: Architektura Obwodu i System Checkpointów (`big_bank_3)Warstwy`)

 — **Ślepa droga**

* Funkcję generującą ansatz rozbudowano o parametr `reps`, pozwalając na powielanie warstw splątania. Badano warianty z 1, 2 oraz 3 warstwami w poszukiwaniu momentu, gdzie szum fizyczny (decoherence) przykrywa korzyści ze zwiększonej nieliniowości.
* Ze względu na bardzo długi czas oczekiwania na wyniki z QPU (timeouty), wprowadzono solidny mechanizm odzyskiwania stanu (`qpu_checkpoint.pkl`). System co 50 zadań zapisuje pobrane paczki (batches) na dysk, umożliwiając bezpieczne wznowienie pętli po błędzie API.

## Etap 4: Sztywne Skalowanie Causalne i Weryfikacja Statystyczna (`big_bank_4)TopologiaGwiazdy`)



* Aby zredukować błędy fizyczne wynikające z bramek SWAP, zmodyfikowano ansatz tak, aby wszystkie połączenia wychodziły z Kubitu 1 (HUB), do którego przypisano kluczową dla modelu parę (Pekao + Obligacje 10Y).
* Wdrożono zewnętrzne mapowanie kwantowe ze sztywnymi, narzuconymi z góry granicami transformacji (`lower_bounds` i `upper_bounds`), co ostatecznie rozwiązało problem look-ahead bias w skalowaniu i zmapowało cechy na sferę od $0$ do $2\pi$.
* Rozszerzono pętlę walidacyjną o Nested Time-Series CV z `GridSearchCV` na zewnętrznym i wewnętrznym etapie.
* Uwiarygodniono wyniki testem istotności statystycznej McNemara oraz generowaniem 95% przedziałów ufności dla AUC metodą Bootstrap.

## Etap 5: Misja Ratunkowa dla Drzew Decyzyjnych (`big_bank_5)XGBoost`)

 — **Ślepa droga**

* Surowe rzuty ortogonalne $X, Y, Z$ generowane przez QPU okazały się kłopotliwe dla klasyfikatora XGBoost, dlatego zaprojektowano transformację `raw_to_spherical`.
* Model zaczął przekształcać rzuty fizyczne na współrzędne sferyczne: promień $r$ (wskaźnik dekoherencji), kąt polarny $\theta$ oraz kąt azymutalny $\phi$.
* W połączeniu z restrykcyjną regularyzacją (`reg_lambda=50`, `max_depth=1`) pozwoliło to XGBoostowi wycinać sensowne reguły decyzyjne z danych sprzętowych.

## Etap 6: Procesy Gaussowskie (`big_bank_6)GPC`)

 — **Ślepa droga**

* Zamieniono Support Vector Machine na `GaussianProcessClassifier` wykorzystujący jądro RBF.
* Przesunięcie skupiło się na Bayesowskiej naturze GPC, która lepiej radzi sobie z wygładzaniem szumu fizycznego pochodzącego z procesora Odra podczas generowania wysoce pewnych map prawdopodobieństw.
* Przetestowano wariant hybrydowy łączący kwantowy QPU SVM, klasyczny Random Forest i klasyczny GPC z jądrem Matern za pomocą "Soft Voting", tworząc ostateczny komitet decyzyjny odporny na pojedyncze błędy.

## Etap 7: Odporność Potoku i Probabilistyka (`big_bank_7)Minipoprawki_Do_Topologii`)



* Zabezpieczono łączenie z chmurą – skrypt łapie wyjątki API i w razie krytycznej awarii automatycznie przerzuca obciążenie transpilacji na awaryjny `AerSimulator`.
* Odświeżono podejście Walk-Forward, wracając do Support Vector Machine, ale precyzyjnie wykorzystując generowanie gładkich prawdopodobieństw z jądra RBF (Platt Scaling) zamiast twardych granic decyzyjnych.

## Etap 8: Algorytmy Genetyczne i Feature Map Optimization (`big_bank_8)Dobór_Wag`)



* Wprowadzono algorytm genetyczny (`scipy.optimize.differential_evolution`), który w sposób zautomatyzowany poszukuje idealnych mnożników wag (od $0.0$ do $1.0$) dla czterech zewnętrznych kubitów topologii gwiazdy.
* Po wyliczeniu najkorzystniejszej Feature Mapy system przesyła zmodyfikowany wektor na QPU, a następnie trenuje klasyfikator.
* Dodano w locie optymalizację progu decyzyjnego, co wyciska maksymalny poziom wskaźnika Accuracy w zmiennych środowiskach rynkowych.
