Głównym zadaniem programu było wykrycie tablicy rejestracyjnej na zdjęciu oraz rozpoznanie jej numerów rejestracyjnych.
W algorytmie skupiono się na odpowiedniej filtracji zdjęcia, wykryciu krawędzi liter a następnie rozpoznaniu ich za pomocą wytrenowanego modelu klasyfikatora.

Na początku zdjęcie jest zmniejszane o 70% oraz skracane o stałą liczbę pixeli w celu wycentrowania tablicy.
Następnie na zdjęcie nakłądany jest filtr medianowy oraz bilateralny w calu odfiltrowania zbędnych szczegółów takich jak: drobne kamienie na drodze, szczegóły maski samochodu lub tyłu, zanieczyszczenia na aucie. 
Jest to bardzo ważny punkt, ponieważ w następnym kroku wykrywane są krawędzie metodą Canny, a na ich podstawie znajdywane są kontury na zdjęciu. 
Po głównej filtracji algorytm nakłada na kontury prostokąty ograniczające (Bounding boxy), sprawdza i zapisuje w tablicy jedynie te kontury, które spełniają założenie, że są literami na tablicy. 
Zakłada się, że dla każdego konturu: szerokość litery jest mniejsza niż wysokość, pole Bboxa jest mniejsze niż 1000 pixeli, wysokość litery > 65 px oraz pole powierzchni konturu > 750 px.
Takie warunki pozwalają zapisać w tabeli wyłącznie kontury i Bboxy liter na tablicy. 

Ponieważ tablice na zdjęciu mogą być ustawione pod kątem do 45 stopni, w celu poprawnej identyfikacji liter należy je wyprosotwać. 
Prostowanie odbywa się za pomocą znalezienia 4 charakterystycznych puntów: lewy górny oraz dolny narożnik Bboxa 1 litery w tablicy oraz prawy górny i dolny narożnik Bboxa ostatniej litery w tablicy.
Zapisywane są współrzędne puntów poiwiększone o 15 pixeli z każdej strony tak, aby po wyprostowaniu widoczna była wyłącznie tablica rejestracyjna. 
W przypadku gdyby na zdjęciu nie wykryto wystarczojącej liczby konturów, prostowanie zdjęcia ograniczone jest do zmniejszenia jego wymiarów.

Wyprostowane i zmniejszone zdjęcie może zawierać krzywe prostokąty ograniczające, dlatego aby poprawić skuteczność wykrywania liter proces został powtórzony.
Nakładane są filtry, wykrywane są krawędzie a następnie kontury, jednak w tym wypadku wyłącznie zewnętrzne, aby nie występował problem z konturami wewnątrz znaków np: "8", "D" lub "O".
Jeśli nie zostaną wykryte litery, zdjęcie jest zmniejszane oraz poddawane operacji morfologicznej domknięcia. 

Na znalezionych literach (max 8 liter) nakładane są Bounding boxy spełniające założenia co do wielkości i rozmiarów.
Każde wykryte kontury sprawdzane są maską na kolor niebieski, aby uniknąć znajdywania 1 konturu jako niebieski prostokąt na początku tablicy.
Jeśli stosunek niebieskiego jest poniżej 25%, normalizowane są rozmiary litery do ujebnoliconej wartości 256x128.
Znormalizowana litera zamieniana jest na wektor cech obrazowych metodą Histogram of Oriented Gradients.
Na podstawie wektorów HOG wytrenowany został model klasyfikatora, który wykonuje predykcję oraz zwraca odpowiednią literę.
Dla pierwszych dwóch konturów w tablicy wytrenowano model z samymi literami, z kolei dla pozostałych liter w tablicy wytrenowano inny model, w którym znajdują się liczby oraz litery z wykluczeniem "O", "D", "Z", "B", "I"
(te litery nie występują w wyróżniku pojazdu). Ostatecznie litery zapisywane są w liście, która zwraca Numer tablicy rejestracyjnej jako tekst i zapisuje go do pliku .json.

Do wytrenowania modeli stworzono bazę ponad 100 własnych zdjęć, które zostały poddane algorytmowi, znalezione litery zostały oznaczone i zapisane do pliku .csv. 
Zastosowanie tego podejścia pozwoliło na bardzo dokładne rozróżnienie liter występujących na tablicach rejestracyjnych w Polsce (skuteczność ok. 98%).
Jako model wybrano MLPClasiffier, w którym dane wektorów poddano normalizacji metodą MinMaxScaler i StandardScaler. 

W przypadku gdyby żadna z wyżej wymienionych metod nie zadziałała, aby uniknąć błędów zakodowano numer pojazdu "PO70153", co jest wartością losową.
