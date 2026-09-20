## Etap 1: Inżynieria Danych i Pierwszy Prototyp (`Kod big_bank1)`)

Na tym etapie powstał fundament całego potoku analitycznego.

* Skrypt ładuje historyczne wyceny P/BV pięciu polskich banków, z których buduje syntetyczny benchmark sektorowy WIG-Banki (baza 1000 pkt).


* Obliczana jest 20-dniowa krocząca zmienność indeksu WIG20 (Z-score) oraz wczytywane są dane makroekonomiczne (WIBOR, EURPLN, rentowność obligacji).


* Zbudowano pierwszą wersję 5-kubitowego obwodu w oparciu o technikę Dense Angle Encoding (1 warstwa rotacji $R_y$, $R_z$ i 1 warstwa splątania).


* Skrypt uruchamia ewaluację Out-Of-Sample, zestawiając klasyczną regresję logistyczną z modelem Projected Quantum Kernel (PQK) działającym na idealnym symulatorze oraz fizycznym procesorze IQM Odra, zapisując zrzut sprzętowy do pliku `.npy`.



## Etap 2: Rygor Walidacyjny i Uszczelnienie Potoku (`Kod big_bank2)`)  <span style="color: red;">Ślepa droga</span>


Ewolucja w kierunku eliminacji wycieku danych (data leakage) podczas strojenia modelu.

* Do kodu wprowadzono struktury `Pipeline`, co gwarantuje, że procesy skalowania (np. `MinMaxScaler`) zachodzą ściśle na danych treningowych w każdej iteracji, chroniąc przed "zajrzeniem" w przyszłość.


* Zaimplementowano siatki `RandomizedSearchCV` sprzężone z obiektem `TimeSeriesSplit` do dynamicznego doboru hiperparametrów (np. parametru $C$ oraz $\gamma$ dla jądra RBF), wykorzystując obszar pod krzywą ROC jako główną metrykę optymalizacyjną.


* Okno treningowe (Walk-Forward) wydłużono do 104 tygodni, zapewniając algorytmom klasycznym dłuższą historię uczenia.



## Etap 3: Architektura Obwodu i System Checkpointów (`Kod big_bank3)`) (ślepa droga)

Testowanie wpływu głębokości obwodu kwantowego na jakość predykcji.

* Funkcję generującą ansatz rozbudowano o parametr `reps`, pozwalając na powielanie warstw splątania. Badano warianty z 1, 2 oraz 3 warstwami w poszukiwaniu momentu, gdzie szum fizyczny (decoherence) przykrywa korzyści ze zwiększonej nieliniowości.


* Ze względu na bardzo długi czas oczekiwania na wyniki z QPU (timeouty), wprowadzono solidny mechanizm odzyskiwania stanu (`qpu_checkpoint.pkl`). System co 50 zadań zapisuje pobrane paczki (batches) na dysk, umożliwiając bezpieczne wznowienie pętli po błędzie API.



## Etap 4: Sztywne Skalowanie Causalne i Weryfikacja Statystyczna (`Kod big_bank4)`)

Całkowita izolacja predykcji od danych z przyszłości oraz wprowadzenie rygoru akademickiego.

* Wdrożono zewnętrzne mapowanie kwantowe ze sztywnymi, narzuconymi z góry granicami transformacji (`lower_bounds` i `upper_bounds`), co ostatecznie rozwiązało problem look-ahead bias w skalowaniu i zmapowało cechy na sferę od $0$ do $2\pi$.


* Rozszerzono pętlę walidacyjną o Nested Time-Series CV z `GridSearchCV` na zewnętrznym i wewnętrznym etapie.


* Uwiarygodniono wyniki testem istotności statystycznej McNemara oraz generowaniem 95% przedziałów ufności dla AUC metodą Bootstrap.



---

**Topologia Gwiazdy i Ablation Study (Etap Przejściowy)**
Pomiędzy wersjami wdrożono kluczową architektoniczną zmianę – *Star Topology*. Aby zredukować błędy fizyczne wynikające z bramek SWAP, zmodyfikowano ansatz tak, aby wszystkie połączenia wychodziły z Kubitu 1 (HUB), do którego przypisano kluczową dla modelu parę (Pekao + Obligacje 10Y). Przeprowadzono metodą *Ablation Study* "uszkadzanie" kolejnych kubitów, badając, jak wyłączenie danego obszaru wpływa na ostateczny spadek AUC. Zaimplementowano również Causal Rolling MinMax (normalizację kroczącą z okna 52 tygodni).

---

## Etap 5: Misja Ratunkowa dla Drzew Decyzyjnych (`Kod big_bank5)`) (ślepa droga)

Eksperymentowanie z klasyfikatorami opartymi o drzewa w przestrzeni kwantowej.

* Surowe rzuty ortogonalne $X, Y, Z$ generowane przez QPU okazały się kłopotliwe dla klasyfikatora XGBoost, dlatego zaprojektowano transformację `raw_to_spherical`.


* Model zaczął przekształcać rzuty fizyczne na współrzędne sferyczne: promień $r$ (wskaźnik dekoherencji), kąt polarny $\theta$ oraz kąt azymutalny $\phi$.


* W połączeniu z restrykcyjną regularyzacją (`reg_lambda=50`, `max_depth=1`) pozwoliło to XGBoostowi wycinać sensowne reguły decyzyjne z danych sprzętowych.



## Etap 6: Procesy Gaussowskie (`Kod big_bank6)`) (ślepa droga)

Eksploracja modeli probabilistycznych opartych na topologii gwiazdy.

* Zamieniono Support Vector Machine na `GaussianProcessClassifier` wykorzystujący jądro RBF.


* Przesunięcie skupiło się na Bayesowskiej naturze GPC, która lepiej radzi sobie z wygładzaniem szumu fizycznego pochodzącego z procesora Odra podczas generowania wysoce pewnych map prawdopodobieństw.


* Następujący po tym wariant hybrydowy (kolejny snippet) połączył kwantowy QPU SVM, klasyczny Random Forest i klasyczny GPC z jądrem Matern za pomocą "Soft Voting", tworząc ostateczny komitet decyzyjny odporny na pojedyncze błędy (50% wagi dla QPU, po 25% dla klasyki).



## Etap 7: Odporność Potoku i Probabilistyka (`Kod big_bank7)`)

Refaktoryzacja systemu nastawiona na maksymalną stabilność wykonania.

* Zabezpieczono łączenie z chmurą – skrypt łapie wyjątki API i w razie krytycznej awarii automatycznie przerzuca obciążenie transpilacji na awaryjny `AerSimulator`.


* Odświeżono podejście Walk-Forward, wracając do Support Vector Machine, ale precyzyjnie wykorzystując generowanie gładkich prawdopodobieństw z jądra RBF (Platt Scaling) zamiast twardych granic decyzyjnych.



## Etap 8: Algorytmy Genetyczne i Feature Map Optimization (`Kod big_bank8)`)

Ostateczna ewolucja w postaci dynamicznego poszukiwania globalnego ekstremum.

* Wprowadzono algorytm genetyczny (`scipy.optimize.differential_evolution`), który w sposób zautomatyzowany poszukuje idealnych mnożników wag (od $0.0$ do $1.0$) dla czterech zewnętrznych kubitów topologii gwiazdy.


* Po wyliczeniu najkorzystniejszej Feature Mapy system przesyła zmodyfikowany wektor na QPU, a następnie trenuje klasyfikator. Dodano w locie optymalizację progu decyzyjnego, co wyciska maksymalny poziom wskaźnika Accuracy w zmiennych środowiskach rynkowych.
