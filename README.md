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

Pisz mi na bieżąco co zrobiłeś i jak wpadniesz na pomysł co dalej robić to daj znać:)

Odpoczywaj i pij wodę kurwa ten
