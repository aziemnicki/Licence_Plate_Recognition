#PL 🇵🇱
# Wykrywanie Tablic Rejestracyjnych

Głównym zadaniem programu było wykrycie tablicy rejestracyjnej na zdjęciu oraz rozpoznanie jej numerów rejestracyjnych. W algorytmie skupiono się na odpowiedniej filtracji zdjęcia, wykryciu krawędzi liter, a następnie rozpoznaniu ich za pomocą wytrenowanego modelu klasyfikatora.

Na początku zdjęcie jest zmniejszane o 70% oraz skracane o stałą liczbę pikseli w celu wycentrowania tablicy. Następnie na zdjęcie nakładany jest filtr medianowy oraz bilateralny w celu odfiltrowania zbędnych szczegółów, takich jak drobne kamienie na drodze, szczegóły maski samochodu lub tyłu, zanieczyszczenia na aucie. Jest to bardzo ważny punkt, ponieważ w następnym kroku wykrywane są krawędzie metodą Canny, a na ich podstawie znajdywane są kontury na zdjęciu.

Po głównej filtracji algorytm nakłada na kontury prostokąty ograniczające (bounding boxes), sprawdza i zapisuje w tablicy jedynie te kontury, które spełniają założenie, że są literami na tablicy. Zakłada się, że dla każdego konturu: szerokość litery jest mniejsza niż wysokość, pole Bboxa jest mniejsze niż 1000 pikseli, wysokość litery > 65 px oraz pole powierzchni konturu > 750 px. Takie warunki pozwalają zapisać w tabeli wyłącznie kontury i Bboxy liter na tablicy.

Ponieważ tablice na zdjęciu mogą być ustawione pod kątem do 45 stopni, w celu poprawnej identyfikacji liter należy je wyprostować. Prostowanie odbywa się za pomocą znalezienia 4 charakterystycznych punktów: lewy górny oraz dolny narożnik Bboxa 1 litery w tablicy oraz prawy górny i dolny narożnik Bboxa ostatniej litery w tablicy. Zapisywane są współrzędne punktów powiększone o 15 pikseli z każdej strony tak, aby po wyprostowaniu widoczna była wyłącznie tablica rejestracyjna. W przypadku gdyby na zdjęciu nie wykryto wystarczającej liczby konturów, prostowanie zdjęcia ograniczone jest do zmniejszenia jego wymiarów.

Wyprostowane i zmniejszone zdjęcie może zawierać krzywe prostokąty ograniczające, dlatego aby poprawić skuteczność wykrywania liter proces został powtórzony. Nakładane są filtry, wykrywane są krawędzie, a następnie kontury, jednak w tym wypadku wyłącznie zewnętrzne, aby nie występował problem z konturami wewnątrz znaków np: "8", "D" lub "O". Jeśli nie zostaną wykryte litery, zdjęcie jest zmniejszane oraz poddawane operacji morfologicznej domknięcia.

Na znalezionych literach (max 8 liter) nakładane są bounding boxy spełniające założenia co do wielkości i rozmiarów. Każde wykryte kontury sprawdzane są maską na kolor niebieski, aby uniknąć znajdywania 1 konturu jako niebieski prostokąt na początku tablicy. Jeśli stosunek niebieskiego jest poniżej 25%, normalizowane są rozmiary litery do ujednoliconej wartości 256x128. Znormalizowana litera zamieniana jest na wektor cech obrazowych metodą Histogram of Oriented Gradients. Na podstawie wektorów HOG wytrenowany został model klasyfikatora, który wykonuje predykcję oraz zwraca odpowiednią literę.

Dla pierwszych dwóch konturów w tablicy wytrenowano model z samymi literami, z kolei dla pozostałych liter w tablicy wytrenowano inny model, w którym znajdują się liczby oraz litery z wykluczeniem "O", "D", "Z", "B", "I" (te litery nie występują w wyróżniku pojazdu). Ostatecznie litery zapisywane są w liście, która zwraca numer tablicy rejestracyjnej jako tekst i zapisuje go do pliku .json.

Do wytrenowania modeli stworzono bazę ponad 100 własnych zdjęć, które zostały poddane algorytmowi, znalezione litery zostały oznaczone i zapisane do pliku .csv. Zastosowanie tego podejścia pozwoliło na bardzo dokładne rozróżnienie liter występujących na tablicach rejestracyjnych w Polsce (skuteczność ok. 98%). Jako model wybrano MLPClassifier, w którym dane wektorów poddano normalizacji metodą MinMaxScaler i StandardScaler.

W przypadku gdyby żadna z wyżej wymienionych metod nie zadziałała, aby uniknąć błędów zakodowano numer pojazdu "PO70153", co jest wartością losową.

#ENG 🏴󠁧󠁢󠁥󠁮󠁧󠁿

# License Plate Detection

The main task of the program was to detect the license plate in an image and recognize its registration numbers. The algorithm focused on proper image filtering, edge detection of the letters, and then recognizing them using a trained classifier model.

Initially, the image is reduced by 70% and cropped by a fixed number of pixels to center the plate. Next, a median and bilateral filter is applied to the image to filter out unnecessary details such as small stones on the road, details of the car's hood or rear, and dirt on the car. This step is crucial because, in the next step, edges are detected using the Canny method, and contours are found in the image based on these edges.

After the main filtering, the algorithm applies bounding boxes to the contours, checks, and records only those contours that meet the assumption of being letters on the plate. It is assumed that for each contour: the width of the letter is less than its height, the Bbox area is less than 1000 pixels, the height of the letter > 65 px, and the contour area > 750 px. These conditions allow only the contours and Bboxes of the letters on the plate to be recorded in the table.

Since the plates in the image can be tilted up to 45 degrees, the letters need to be straightened for correct identification. Straightening is done by finding 4 characteristic points: the upper left and lower corner of the Bbox of the first letter on the plate, and the upper right and lower corner of the Bbox of the last letter on the plate. The coordinates of the points are recorded, increased by 15 pixels on each side so that only the license plate is visible after straightening. If not enough contours are detected in the image, straightening is limited to reducing its dimensions.

The straightened and reduced image may contain skewed bounding boxes, so to improve letter detection effectiveness, the process is repeated. Filters are applied, edges are detected, and then contours, but in this case only external ones, to avoid problems with contours inside characters like "8", "D", or "O". If no letters are detected, the image is reduced and subjected to a morphological closing operation.

Bounding boxes are applied to the detected letters (max 8 letters) that meet the size and dimension assumptions. Each detected contour is checked with a blue color mask to avoid finding one contour as a blue rectangle at the beginning of the plate. If the blue ratio is below 25%, the letter sizes are normalized to a uniform value of 256x128. The normalized letter is converted into a feature vector using the Histogram of Oriented Gradients method. Based on the HOG vectors, a classifier model is trained, which makes predictions and returns the appropriate letter.

For the first two contours on the plate, a model was trained with only letters, while for the remaining letters on the plate, another model was trained, which includes numbers and letters excluding "O", "D", "Z", "B", "I" (these letters do not appear in the vehicle identifier). Ultimately, the letters are recorded in a list that returns the license plate number as text and saves it to a .json file.

To train the models, a database of over 100 own photos was created, which were processed by the algorithm, the detected letters were marked, and saved to a .csv file. This approach allowed for very accurate differentiation of letters on license plates in Poland (accuracy approx. 98%). The MLPClassifier model was chosen, in which the vector data was normalized using the MinMaxScaler and StandardScaler methods.

In case none of the above methods worked, to avoid errors, the vehicle number "PO70153" was encoded, which is a random value.
