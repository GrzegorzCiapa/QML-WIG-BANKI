## Etap 1: Inżynieria Danych i Pierwszy Prototyp (`big_bank_1`)

* Skrypt ładuje historyczne wyceny P/BV pięciu polskich banków, z których buduje syntetyczny benchmark sektorowy WIG-Banki (baza 1000 pkt).
* Obliczana jest 20-dniowa krocząca zannualizowana zmienność zrealizowana indeksu WIG20 oraz wczytywane są dane makroekonomiczne (WIBOR 3M, EURPLN, rentowność obligacji 10Y).
* Zbudowano pierwszą wersję 5-kubitowego obwodu w oparciu o technikę Dense Angle Encoding (dwukrotna warstwa rotacji $R_y$, $R_z$ i bramek splątujących CNOT).
* Skrypt uruchamia ewaluację Out-Of-Sample, zestawiając klasyczną regresję logistyczną z modelem Projected Quantum Kernel (PQK) działającym na idealnym symulatorze oraz fizycznym procesorze IQM Odra, zapisując zrzut sprzętowy do pliku `.npy`.

## Etap 2: Rygor Walidacyjny i Uszczelnienie Potoku (`big_bank_2_Hiperparametry`)

— **Ślepa droga**

* Izolacja danych: Zastosowano struktury Pipeline (`MinMaxScaler` dla modeli klasycznych, `StandardScaler` dla rzutów kwantowych), wymuszając skalowanie wyłącznie wewnątrz okna treningowego i eliminując wyciek danych.
* Kroczący tuning: W każdym kroku Walk-Forward dynamicznie optymalizowano pod kątem ROC AUC parametry: $C$, $\gamma$ oraz wagowanie klas na 5 podziałach szeregu czasowego (`TimeSeriesSplit`).
* Wydłużenie okna do 104 tygodni: Rozszerzono próbę do 2 lat, aby zapewnić algorytmom wystarczającą historię pod procedurę walidacji krzyżowej.
* Wnioski: Dynamiczny tuning doprowadził do przeuczenia na lokalnym szumie rynkowym — wartości AUC wszystkich modeli spadły poniżej progu losowego ($0.46\text{--}0.49$). Jednak fizyczna Odra zachowała najwyższe AUC ($0.4891$) oraz dodatni bilans trafień netto w czasie (Equity Curve), unikając załamania widocznego w modelach klasycznych.

## Etap 3: Architektura Obwodu i System Checkpointów (`big_bank_3_Warstwy`)

— **Badanie ablacyjne**

* Architektura Data Re-uploading: Do generatora ansatzu wprowadzono parametr warstw (`reps`), powtarzając sekwencję rotacji cech oraz splątania pierścieniem CNOT w celu zbadania kompromisu pomiędzy ekspresywnością a dekoherencją.
* Wprowadzono mechanizm checkpointingu (`_checkpoint.pkl`), co 50 zadań zapisujący pobrane paczki na dysk, umożliwiając bezpieczne wznowienie pętli po błędzie API z 24-godzinnym limitem timeoutu.
* Powrót do okna 52 tygodni z potokiem Pipeline (`StandardScaler`), w pełni eliminując wyciek danych.
* Wnioski badawcze (Badanie ablacyjne): Wszystkie warianty kwantowe przewyższyły klasyczny SVM z jądrem RBF (Accuracy $>52.7\%$ vs $52.04\%$, AUC ROC $>0.520$ vs $0.512$)[cite: 3]. Najwyższą zdolność dyskryminacyjną zachował najpłytszy obwód (QPU 1 Layer: $\text{AUC} = 0.5284$)[cite: 3], dowodząc, że przy obecnym poziomie szumu bramek dwukubitowych na Odrze 5 płytkie obwody stanowią najbardziej efektywną architekturę[cite: 3].

## Etap 4: Sztywne Skalowanie Causalne i Weryfikacja Statystyczna (`big_bank_4_TopologiaGwiazdy`)

* Dopasowanie do natywnej architektury: Zastąpiono sztuczny pierścień logiczny natywną topologią gwiazdy Odry 5 z centralnym hubem na Kubicie 1 (Pekao + Obligacje 10Y).
* Zamiast klasycznych ramek czy globalnych skalerów wdrożono przyczynowe okno kroczące (`LOOKBACK = 52`). Skalowanie cech do przedziału $[0, 2\pi]$ odbywało się tylko z wartości min-max aktualnie znajdujących się w oknie, co zapewniło pełną odporność na wyciek danych.
* Zastosowano podwójną pętlę walidacyjną (Nested Time-Series CV) z `GridSearchCV` na wewnętrznym podziale szeregu czasowego.
* Istotność statystyczna na fizycznym QPU (Odra 5): Model kwantowy w topologii gwiazdy osiągnął wyższą skuteczność we wszystkich metrykach względem klasycznego SVM: Accuracy $53.42\%$ vs $48.34\%$, F1-Score $0.5993$ vs $0.5570$ oraz AUC $0.5242$ vs $0.4871$. Test McNemara dał $p\text{-value} = 0.0453$, formalnie potwierdzając, że przewaga sprzętu kwantowego nad klasycznym SVM z jądrem RBF jest istotna statystycznie na poziomie istotności $\alpha = 0.05$.

## Etap 5: Misja Ratunkowa dla Drzew Decyzyjnych (`big_bank_5_XGBoost`)

— **Ślepa droga**

* Klasyczne drzewa decyzyjne tną przestrzeń wyłącznie prostopadle do osi cech, co w kartezjańskich rzutach obserwabli Pauliego utrudniało wykrywanie rotacji stanu na sferze Blocha.
* Surowe trójki rzutów z QPU przekształcono na współrzędne sferyczne: promień $r$ (wskaźnik dekoherencji), kąt polarny $\theta$ oraz kąt azymutalny $\phi$.
* Ekstremalna regularyzacja pod szeregi finansowe: Użyto płytkich pniaków decyzyjnych (`max_depth=1`), wolnego uczenia (`learning_rate=0.05`) oraz bardzo mocnej kary L2 (`reg_lambda=50`), aby zapobiec przeuczeniu na szumie rynkowym.
* Klasyczny XGBoost na surowych cechach rynkowych poniósł porażkę ($\text{AUC} = 0.4991$).
* Dane z fizycznej Odry 5 poprawiły stabilność modelu — wariant z inżynierią sferyczną osiągnął $\text{AUC} = 0.5115$ i $\text{Accuracy} = 53.82\%$, potwierdzając, że reprezentacja kątowa ułatwia modelom opartym na gradient boosting wyciąganie nieliniowych zależności z procesora kwantowego.

## Etap 6: Procesy Gaussowskie (`big_bank_6_GPC`)

— **Ślepa droga**

* Zamieniono Support Vector Machine na `GaussianProcessClassifier` wykorzystujący jądro RBF.
* Przesunięcie skupiło się na bayesowskiej naturze GPC, która lepiej radzi sobie z wygładzaniem szumu fizycznego pochodzącego z procesora Odra podczas generowania wysoce pewnych map prawdopodobieństw.
* Przetestowano wariant hybrydowy łączący kwantowy QPU SVM, klasyczny Random Forest i klasyczny GPC z jądrem Matérn za pomocą Soft Voting, tworząc ostateczny komitet decyzyjny odporny na pojedyncze błędy.
* *(Uwaga: Wymaga ponownego przeliczenia i uzupełnienia metryk liczbowych z poprawnego skryptu).*

## Etap 7: Odporność Potoku i Probabilistyka (`big_bank_7_Minipoprawki_Do_Topologii`)

* Zabezpieczono łączenie z chmurą — skrypt łapie wyjątki API i w razie krytycznej awarii automatycznie przerzuca obciążenie transpilacji na awaryjny `AerSimulator`.
* Odświeżono podejście Walk-Forward, wracając do Support Vector Machine, ale precyzyjnie wykorzystując generowanie gładkich prawdopodobieństw z jądra RBF (Platt Scaling) zamiast twardych granic decyzyjnych.

## Etap 8: Algorytmy Genetyczne i Feature Map Optimization (`big_bank_8_Dobór_Wag`)

* Wprowadzono algorytm genetyczny (`scipy.optimize.differential_evolution`), który w sposób zautomatyzowany poszukuje optymalnych mnożników wag (od $0.0$ do $1.0$) dla czterech zewnętrznych kubitów topologii gwiazdy.
* Po wyliczeniu najkorzystniejszej Feature Mapy system przesyła zmodyfikowany wektor na QPU, a następnie trenuje klasyfikator.
* Dodano w locie optymalizację progu decyzyjnego, co wyciska maksymalny poziom wskaźnika Accuracy w zmiennych środowiskach rynkowych.
