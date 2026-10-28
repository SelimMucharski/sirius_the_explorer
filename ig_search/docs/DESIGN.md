# ig_search — dokument projektowy

Ten dokument opisuje **docelowy** system przeszukiwania oraz różnice względem obecnego kodu.
Kolejność prac i kryteria akceptacji są w `docs/ROADMAP.md`.

---

## 0. Kontekst

- Zadanie: konkurencja typu ERC na Mars Yardzie, łazik Sirius 2.
- W obszarze leżą **M = 3** nieruchome próbki o nieznanych położeniach (na powierzchni lub częściowo wbite).
- Próbkę trzeba **wykryć, podjechać, zatrzymać się i zgłosić** z estymatą położenia leżącą w obszarze roboczym manipulatora.
- Budżet czasu T_max = 30 min; pełna punktacja poniżej 7,5 min, potem schodkami co 5 min do 60 %
  (regulamin ERC 7.3.4.4 a, §6 pkt 41).
- Misja kończy się po **trzech zgłoszeniach** (także fałszywych!) albo po T_max.
- Badania prowadzone są **w symulacji**; poza łazika jest znana.
- Detektor ma **fałszywe alarmy** (np. kamienie mylone z próbką).
- Teren jest **górzysty** — przesłonięcia przez sam teren są kluczowe.

## 1. Notacja (zgodna z pracą dyplomową, podrozdział 2.5)

| Symbol | Znaczenie | Uwagi do kodu |
|---|---|---|
| W ⊂ ℝ² | rzut Mars Yardu na płaszczyznę | |
| a | bok komórki siatki | param `cell_size` |
| N, i = 1..N | liczba komórek, indeks komórki | |
| N_d | liczba komórek dostępnych | |
| q_t = (x_t, y_t, ψ_t) | poza łazika (ψ — yaw) | **w obecnym kodzie `q` oznacza belief — do zmiany** |
| h: W → ℝ | mapa wysokości terenu | nowe |
| M | liczba próbek (= 3) | |
| s_j | prawdziwe położenie próbki j | znane tylko w symulatorze |
| p_i | belief: P(w komórce i jest nieodnaleziona próbka) | w kodzie: `p` |
| p_i⁰ = M / N_d | prior jednorodny | wariant niejednorodny: podrozdz. 6.4 |
| ℓ_i = ln(p_i / (1 − p_i)) | log-szanse | belief przechowywany jako ℓ |
| V(q) | zbiór komórek widocznych z pozy q | |
| R(q) | obszar roboczy manipulatora (wycinek koła, promień r_rob) | |
| ŝ | estymata położenia próbki | |
| t_j | chwila poprawnego zgłoszenia próbki j | |
| P_d(r, α) | P(wykrycie \| próbka w widocznej komórce) | r — odległość, α — kąt od osi kamery |
| σ(r) | odchylenie std. błędu pozycji detekcji | rośnie z r |
| λ_fa | intensywność fałszywych alarmów [1/(m²·klatka)] | niezależna od a |
| ε | tolerancja chwytu | |
| τ_n | próg prawdopodobieństwa dla n-tego zgłoszenia | |
| f(t) | funkcja punktacji w czasie | |

## 2. Stan obecny kodu (punkt wyjścia)

- `ig_search/core.py` — czysty numpy/scipy, bez ROS:
  - siatka beliefu (zgrubna względem mapy), prior jednorodny lub z pliku,
  - widoczność **2D** (wycinek koła, raycasting po mapie zajętości), P_d(r) liniowo malejące,
  - prekomputacja macierzy rzadkiej `V[v, i] = P_d` dla kandydackich póz (siatka × n_yaw),
  - aktualizacja Bayesa tylko dla braku detekcji, **bez fałszywych alarmów**,
  - zysk: `detections` (V @ p) lub `entropy`,
  - czas: Dijkstra po grafie bezpiecznych komórek + obroty (napęd różnicowy) + narzut,
  - wybór zachłanny: argmax zysk / czas.
- `ig_search/search_planner.py` — węzeł ROS 2 Jazzy: `/map`, `/target_detections` (PoseArray), TF, akcja `navigate_to_pose`, histereza, publikacja beliefu.

Ograniczenia, które ten projekt usuwa: brak fałszywych alarmów, model 2D (brak terenu), detekcja = pewne znalezienie,
brak fazy dojazdu i zgłoszenia, zachłanność (niedopasowanie do miary t₃), aktualizacje wnioskowane z ruchu zamiast z klatek.

## 3. Model docelowy

### 3.1 Teren i widoczność 2,5D

- Wejście: mapa wysokości h na siatce (na start: plik `.npy` + `.yaml` z rozdzielczością i originem; później opcjonalnie `grid_map_msgs/GridMap`).
- Pełna poza kamery wyznaczana z q i h: pozycja łazika na terenie (z = h(x, y)), pochylenie/przechył z gradientu h
  w kierunku ψ i prostopadłym, plus stałe zamocowanie kamery (przesunięcie x_c, z_c, pochylenie montażu).
- Komórka i ∈ V(q), jeśli dla punktu powierzchni (c_i, h(c_i) + δ) (δ — wysokość próbki nad terenem):
  1. kąt poziomy względem osi kamery ∈ [−FOV_h/2, FOV_h/2],
  2. kąt pionowy ∈ [−FOV_v/2, FOV_v/2] (daje naturalnie minimalny zasięg r_min),
  3. r ∈ [r_min, r_max],
  4. odcinek kamera → punkt nie przechodzi pod terenem (marsz po odcinku, test wysokości z marginesem).
- Implementacja: dla danej pozy wektorowo po wszystkich komórkach w kole r_max (test per komórka, nie wachlarz promieni).
- Prekomputacja `V[v, i] = P_d(r_i(v), α_i(v))` dla wszystkich kandydackich póz, jak obecnie.

### 3.2 Model detektora (obiekty, nie komórki)

Detektor w każdej przetworzonej klatce zwraca **listę** detekcji ẑ_k (może być pusta). Model:

- próbka w komórce i ∈ V(q) jest wykrywana z prawdopodobieństwem P_d(r_i, α_i),
- pozycja detekcji: ẑ ~ 𝒩(c_i, σ(r_i)² I),
- fałszywe alarmy: proces Poissona o intensywności λ_fa na m² widocznego terenu.

**Aktualizacja (jedna formuła dla obu przypadków).** Dla każdej widocznej komórki i, po klatce ze zbiorem detekcji {ẑ_k}:

```
LR_i = (1 − P_d) + P_d · Σ_k 𝒩(ẑ_k; c_i, σ²) / λ_fa
ℓ_i ← ℓ_i + ln(LR_i)
```

Interpretacja: jeśli w i jest próbka, to albo została wykryta (gęstość P_d·𝒩), albo pominięta, a obecne detekcje są
fałszywymi alarmami ((1 − P_d)·λ_fa); jeśli próbki nie ma, detekcje są fałszywymi alarmami (λ_fa).
Daleko od wszystkich detekcji LR_i → 1 − P_d (zwykły brak detekcji); blisko detekcji LR_i ≫ 1.

Przybliżenia (opisać w pracy): komórki niezależne, brak jawnej asocjacji danych, sumowanie po detekcjach.
Praktycznie: liczyć składnik 𝒩 tylko w promieniu 3σ od ẑ_k; ograniczać ℓ do [ℓ_min, ℓ_max].

**Bramka różnorodności widoku.** Obserwację komórki i łączymy z beliefem tylko, jeśli punkt widzenia różni się od
wszystkich poprzednio użytych dla tej komórki o więcej niż Δ_kier (kierunek) lub Δ_odl (względna odległość).
Implementacja: dla każdej komórki krótka lista (kierunek, odległość) użytych obserwacji lub kubełki kierunków.
Cel: nie „potwierdzać” tego samego kamienia z tego samego miejsca (fałszywe alarmy są skorelowane).

**Źródło obserwacji.** Aktualizacja per **klatka detektora** (z jej znacznikiem czasu i pozą z TF na ten czas),
a nie wnioskowana z ruchu łazika. Pusta lista detekcji też jest wiadomością.

### 3.3 Kandydaci

- Kandydat k = spójna składowa komórek z p_i > p_seed (np. 0,02).
- Masa: P_k = 1 − Π_{i∈k} (1 − p_i) (prawdopodobieństwo ≥ 1 próbki w skupisku).
- Estymata: ŝ_k = Σ p_i c_i / Σ p_i.
- Poza dojazdu q_app(k): poza osiągalna, w której ŝ_k ∈ R(q) (np. ŝ_k w odległości ~r_rob/2 na wprost),
  wybrana spośród kilku kierunków podejścia z minimalnym czasem.

### 3.4 Akcje i wspólna waluta

Wszystkie akcje oceniamy jako **oczekiwaną liczbę poprawnie zgłoszonych próbek na sekundę**.

- Obserwuj z pozy v: `S_obs(v) = Σ_i V[v, i] · p_i / T(v)`
- Podjedź do kandydata k: `S_app(k) = P_k / T_app(k)`

W każdej chwili wykonywana jest akcja o największej wartości (przy horyzoncie — pierwsza akcja najlepszej sekwencji).
Podczas dojazdu belief jest aktualizowany obserwacjami z bliska; jeśli P_k spadnie (kamień), akcja przegrywa i łazik zawraca.
Histereza jak w obecnym kodzie.

### 3.5 Reguła zgłoszenia

W pozie q (po dojeździe) zgłaszamy z estymatą ŝ, jeśli:

```
P_rep = 1 − Π_{i: c_i ∈ R(q) ∩ B(ŝ, ε)} (1 − p_i)  ≥  τ_n
```

gdzie n — numer zgłoszenia (1, 2, 3). Uzasadnienie progu: zgłoszenie opłaca się, gdy τ·G > (1 − τ)·C_n,
czyli τ_n ≥ C_n / (G + C_n); fałszywe trzecie zgłoszenie kończy misję, więc C_3 ≫ C_1 i τ_3 > τ_1.
Domyślnie (do strojenia): τ = [0,80; 0,85; 0,95].
Jeśli P_rep < τ_n: wykonać obserwację weryfikującą z innej strony (z bramką różnorodności) i decydować ponownie.
Po zgłoszeniu: wyzerować belief w R(q) ∩ B(ŝ, ε') i zwiększyć licznik zgłoszeń.

**Otwarte pytanie (do decyzji autora):** czy system dowiaduje się, że zgłoszenie było fałszywe (np. nieudany chwyt)?
Domyślnie: nie — system traktuje zgłoszenie jako odnalezienie.

### 3.6 Horyzont planowania i punktacja

- Beam search po sekwencjach akcji o głębokości D ∈ {1, 2, 3} i szerokości B.
- Symulowana obserwacja: wariant „brak detekcji” (p_i ← p_i(1 − P_d)/(1 − p_i P_d)) — najbardziej prawdopodobny wynik przy małych p.
  Dzięki V to tylko operacja na wektorze.
- Wartość sekwencji: Σ_k f(t_k) · zysk_k, gdzie t_k — przewidywany czas zakończenia k-tej akcji,
  f(t) — znormalizowana punktacja (1 do 7,5 min, potem malejąca; kształt z regulaminu, parametryzowany;
  schodkowa - §6 pkt 41).
- Receding horizon: wykonujemy pierwszą akcję, replanujemy.
- D = 1 i f ≡ 1 odpowiada obecnemu wyborowi zachłannemu.

### 3.7 Czas ruchu po terenie

- Graf 8-sąsiedztwa po komórkach przejezdnych; waga krawędzi = czas:
  `długość / (v_max · g(nachylenie))`, krawędzie o nachyleniu > slope_max usunięte
  (opcjonalnie asymetria pod górę / w dół).
- Obroty jak obecnie: α / ω_max; plus stały narzut t_0.
- Ten sam model w planerze i w symulatorze kinematycznym (w symulatorze z dodatkowym szumem).

## 4. Symulator i eksperymenty

### 4.1 Symulator (bogatszy niż model planera — unikanie „zbrodni odwrotnej”)

- **Świat:** prawdziwa mapa wysokości (model 3D Mars Yardu lub generowany teren), M = 3 próbki losowane na komórkach dostępnych,
  n_rocks kamieni-dystraktorów z „mylnością” c_r.
- **Detektor symulowany:**
  - P_d prawdziwe zależne od r, α, kąta obserwacji względem próbki i stopnia wbicia,
  - błąd pozycji z σ_true(r),
  - fałszywe alarmy od kamieni **trwałe**: ten sam kamień z podobnego punktu widzenia daje ten sam wynik
    (deterministyczny hash kubełka punktu widzenia + seed),
  - słabe tło fałszywych alarmów (Poisson),
  - widoczność z prawdziwego terenu, planer dostaje mapę h z błędem.
- **Ruch:** symulator kinematyczny bez Gazebo (łazik jedzie po ścieżce z grafu z prędkościami z modelu ± szum),
  klatki detektora co Δt. Cel: K ≈ 100 przebiegów w rozsądnym czasie.
- Pełna zgodność z regułami misji: zgłoszenie poprawne, jeśli nieodnaleziona próbka s_j ∈ R(q) i ‖ŝ − s_j‖ < ε.

### 4.2 Strategie (wszystkie z tym samym detektorem, beliefem i regułą zgłoszenia)

1. Meander — rozstaw pasów z szerokości pola widzenia (z zakładką), jak w pracy.
2. Najbliższy nieobejrzany obszar (czyste pokrycie, frontier).
2a. „Chase” - dojazd do każdej surowej detekcji, poza tym frontier (§6 pkt 42).
3. Proponowana, zachłanna (D = 1).
4. Proponowana z horyzontem (D = 2, 3) i f(t).

### 4.3 Miary

- t₁ ≤ t₂ ≤ t₃; t₃ jako dystrybuanta (Kaplan-Meier, cenzurowanie na T_max) + średnia/mediana z rozrzutem,
- η = n_pop / (M·K), π = n_pop / (n_pop + n_fał), L, e_j = ‖ŝ_j − s_j‖,
- średnia punktacja wg f(t),
- przebieg sumarycznej entropii beliefu w czasie (uwaga: nie jest monotoniczna — skoki w górę po detekcjach).
- Te same seedy rozmieszczeń dla wszystkich strategii (porównania parami).

### 4.4 Ablacje

λ_fa; prior jednorodny vs niejednorodny (6.4); bramka różnorodności on/off; głębokość horyzontu D; błąd mapy h; P_d z kątem vs bez.

## 5. Uwagi do sformułowania w pracy (2.5) — uzasadnienie zmian

1. Wprowadzić jawnie h; pełna poza kamery wynika z q_t i h.
2. Rozróżnić niezależność położeń próbek od niezależności komórek (to drugie to przybliżenie: P ≈ M/N_d przy N_d ≫ M).
3. Fałszywe alarmy opisać intensywnością na powierzchnię (λ_fa), nie prawdopodobieństwem na komórkę.
4. Kryterium poprawności zgłoszenia uzupełnić o ‖ŝ − s_j‖ < ε.
5. Decyzję o zgłoszeniu nazwać jako osobny problem decyzyjny (progi τ_n).
6. Zdefiniować sposób liczenia t₃ dla przebiegów nieukończonych (cenzurowanie).
7. Omówić niedopasowanie zachłannego kryterium do t₃ (motywacja horyzontu i f(t)).
8. Symulowany detektor musi być bogatszy niż model planera.

## 6. Zmiany modelu wprowadzone przy implementacji (M3–M4, 2026-10-19)

Wykryte testami misji offline; każda ma test regresyjny w `test/`. Do opisania w pracy (rozdz. 4).

1. **Mianownik aktualizacji z wyjaśnianiem przez inne źródła (§3.2).** Zamiast `λ_fa`:

   ```
   LR_i = (1 − P_d,i) + P_d,i · Σ_k 𝒩_ik / (λ_fa + Σ_{j≠i} P_d,j 𝒩_jk p_j)
   ```

   Bez tego przy małym λ_fa jedna próbka czyni „pewnymi” wiele komórek w promieniu ~0,5 m
   (test: masa > 3 dla jednej próbki), a po wyzerowaniu otoczenia zgłoszonej próbki resztka
   wygląda jak kolejna próbka - w teście misji ta sama próbka była zgłaszana dwa razy.
   Z wykluczeniem komórki i z mianownika: masa skupiska ≈ 1, sąsiedzi maleją.
   Wykluczenie samej komórki jest istotne - bez niego p nasyca się w okolicy 0,5.
   Odrzucony wariant: reprezentacja intensywności (filtr PHD, D_i = ln⁻¹ ℓ_i) - pojedyncze
   pominięcie dobrze potwierdzonej próbki zbijało D z ~1,2 do ~0,2 i kandydat „znikał”.
   Parametr `belief.explain_away` (ablacja: `false` = formuła z §3.2).
2. **Gęstość uśredniona po komórce:** σ_eff² = σ(r)² + a²/12. Przy σ ≪ a (nadgarstek, σ ≈ 2 cm,
   a = 20 cm) detekcja próbki z dala od środka komórki nie trafiała w żadną komórkę i działała
   jak obserwacja negatywna.
3. **Zysk obserwacji spójny z bramką różnorodności (§3.4).** V liczone tylko dla par (v, i),
   w których punkt widzenia v jest dla komórki i nowy; inaczej łazik stał w pozie i „obserwował”
   w kółko (zakleszczenie). Zysk obserwacji z p̃ = p poza skupiskami kandydatów (kandydatów
   obsługuje akcja dojazdu - brak podwójnego liczenia). Kubełki kierunku wyśrodkowane na
   wielokrotnościach 45° (kursy póz kandydackich są kwantowane tak samo).
4. **Waluta obserwacji:** `S_obs(v) = Σ V p̃ / (T(v) + T̂_app(v))`, T̂_app - oczekiwany czas od
   detekcji do zgłoszenia (dojazd z v do wykrytej komórki + t_verify); obie akcje w zgłoszeniach/s.
5. **Po zgłoszeniu:** komórki w ε' = 0,4 m trwale wyłączone (odpowiada podjęciu próbki - w symulacji
   próbka zostaje na miejscu); resztki skupiska poza ε' wracają do prioru; detekcje w promieniu
   ε' + 2σ(r) od zgłoszonych próbek są do nich przypisywane i pomijane.
6. **Porzucony kandydat** jest ignorowany przez `block_time` = 90 s, chyba że P_k wzrośnie o 0,2
   (najwyżej do 0,99) - wcześniej blokada bywała dożywotnia.
7. **ŝ w zgłoszeniu** z detekcji kamery na nadgarstku, jeśli jest w promieniu 0,3 m (σ ≈ 2 cm).
8. **Mapa wysokości** z wizualnego mesha terenu (140 tys. trójkątów, 0,1 m); kolizyjny jest
   ~20× rzadszy (zgodność p95 4 cm). Chropowatość = odch. std. od średniej lokalnej (bez trendu
   zbocza); braki danych na brzegu wypełniane najbliższą wysokością.
9. **Pole widzenia** testowane w płaszczyźnie obrazu (rzut perspektywiczny), nie w kątach sferycznych.
10. **Dojazd i chwyt (2026-10-20, decyzja autora).** Przed chwytem chwytak jest dokładnie
    pionowo nad próbką (oś podejścia = pion w świecie, także na zboczu; palce na pionie przez
    środek próbki) - bez podejść odchylonych. Dojazd nie ustawia łazika centralnie przed próbką:
    kończy się, gdy tylko ŝ jest w R(q) z zapasem (0,035 m, 10°), także w trakcie jazdy Nav2;
    korekta końcowa obraca łazik tylko do granicy zakresu kątów i jedzie prosto (obrót w miejscu
    napędem burtowym przesuwa łazik, na zboczu do 0,3 m). Ponowne podejście jedzie do nowej pozy
    (inny punkt widzenia), ale bez obrotu do kursu; podjazd w zasięg po weryfikacji - krótka
    korekta z bieżącej pozy, bez Nav2.
11. **R(q) zależny od terenu (2026-10-20, decyzja autora).** R(q) z IK dla pełnego chwytu
    (pozycja przed chwytem + opuszczenie palców 12 mm poniżej osi próbki): pierścień wokół
    podstawy ramienia (0,1145 m przed base_link), ±75° od kursu, r ∈ [0,60; r_max(dz)], gdzie
    dz - wysokość środka próbki względem base_link z mapy wysokości, r_max interpolowany
    z tabeli IK: dz = −6/−3/0/+1,5/+3/+6 cm → 0,62/0,68/0,72/0,74/0,76/0,78 m. Poza dojazdu
    0,78 m (środek pierścienia na płaskim terenie).
12. **Chwyt i zebranie próbki (2026-10-20, decyzja autora).** Po pozycji przed chwytem pomiar
    z bliska (mediana detekcji nadgarstka z nieruchomego ramienia; korekta pozycji przy różnicy
    > 1 cm), pionowe opuszczenie wprost do kontrolera ramienia, zamknięcie chwytaka, podniesienie;
    chwyt udany, gdy rozstaw szczęk ≈ średnica próbki. Ruch z próbką 20% prędkości. W pozycji
    transportowej próbka jest zebrana (w symulacji usuwana z Gazebo). Pozycja do weryfikacji
    i chwytu - mediana detekcji z nieruchomej kamery nadgarstka (detekcje z ruchu ramienia
    myliły się do 6 cm).
13. **Kalibracja detektora w Gazebo (M6, 2026-10-20).** Tablice w `config/calib/` (format
    DetectorModel.load_table), zbierane narzędziem `ig_search_experiments/calib_collect`: dla każdej
    klatki V(q) z przesłanianiem dla prawdziwej pozy kamery (Gazebo + TF), r, α, wykrycie, błąd 2D.
    Kamera przednia (1787 klatek, 99 par poza-próbka): P_d = 1,0 do 2,5 m, 0,93 / 0,87 / 0,67 /
    0,60 / 0,14 dla 2,5-3 / 3-3,5 / 3,5-4 / 4-5 / 5-6 m (bez podziału na α - pominięcia przychodzą
    całymi pozami, np. próbka widziana końcem); σ 0,03 m (< 1 m) do 0,35 m (5-6 m), głównie
    obciążenie odległości (detektor zaniża ją o ~7%; w poprzek promienia ~1 cm); λ_fa =
    0,0064 /(m²·klatka) (0,13 FA na klatkę). Nadgarstek (ramię nieruchome, 33 pary): P_d
    0,73-1,0 (niższe przy osi kadru - szczęki chwytaka), σ 0,7-2,3 cm, λ_fa = 0,2 /(m²·klatka)
    (FA 0,15-0,75 m od próbki - prawdopodobnie szczęki; podwójne detekcje próbki < 0,15 m
    niewliczone). σ wygładzone do niemalejącej z r. Ruch ramienia nie pogarsza detekcji
    nadgarstka, gdy poza łazika jest dokładna.
14. **Chwyt w ramce odometrii (2026-10-20).** Błędy 3-8 cm pozycji do chwytu w misjach brały się
    z lokalizacji, nie z detektora (mediana błędu nadgarstka przy pozie z Gazebo 1,5 cm): pomiar
    przeliczany do `map` przez EKF+GNSS w chwili klatki, a chwyt do `base_link` w chwili późniejszej
    - skok korekty GNSS przenosił się na pozycję palców. Detekcje nadgarstka niosą też pozycję
    w `odom` (gładka, bez GNSS); adapter ramienia przelicza ją przez `odom -> base_link`.
15. **Logika misji jako drzewo zachowań (2026-10-25, decyzja autora).** Automat stanów
    `Mission` zastąpiony drzewem py_trees (`application/mission_tree.py`), tykanym raz na
    `Mission.step()`. `Mission` zostaje kontekstem (belief, kandydat, zgłoszenia) z operacjami
    wołanymi przez węzły; interfejs dla węzła ROS i numeracja `MissionStatus` bez zmian.
    Drzewo (ok. 20 węzłów): Koniec misji? (M zgłoszeń, brak akcji, T_max) | Kandydat: Śledzony
    kandydat? → [Kandydat → zgłoszenie: Dojazd, Skan weryfikujący, P_rep ≥ τ_n?,
    Zasięg (ŝ ∈ R(q)? | Podjazd w zasięg), Zgłoszenie (Przed chwytem z jedną korektą, Chwyt,
    Publikacja)] | Podejście z innej strony (≤ max_verify_attempts) | Porzucenie] |
    Przeszukiwanie: Wynik obserwacji → Planowanie. Składanie ramienia przed jazdą i odłożony
    cel nawigacji są wewnątrz akcji jazdy; porażka pozycji przed chwytem albo chwytu nie
    przerywa zgłoszenia (flagi). Wariant bez nadgarstka: drzewo budowane bez skanu i chwytu
    (ustalenie się kamery przedniej, potem podjazd w zasięg i publikacja). Fazy
    SEARCH/APPROACH/VERIFY/REPORT - tylko etykieta statusu i filtra klatek nadgarstka.
    **Zmiana zachowania:** po podjeździe w zasięg nie ma ponownej weryfikacji (wcześniej skan
    jeszcze raz) - pozycję do chwytu i tak poprawia pomiar z bliska przed chwytem; nieudany
    podjazd prowadzi do podejścia z innej strony. Na 27 przebiegach atrap (3 strategie,
    ziarna 0/3/7, z nadgarstkiem i bez) 78/78 poprawnych zgłoszeń wobec 70/70 automatu,
    łączny czas −14%, porzuceń 42 wobec 60 - tylko kontrola regresji, nie ocena (detektor
    testowy).
16. **[Wycofane 2026-10-25, zob. pkt 29] Ponowna reguła zgłoszenia z bliska (2026-10-25).** W gałęzi Zgłoszenie
    (z nadgarstkiem) po pozycji przed chwytem, a przed chwytem, warunek P_rep ≥ τ_n liczony
    ponownie na bieżącym beliefie (klatki nadgarstka z ~0,3 m - nowy kubełek odległości bramki
    różnorodności). Nie przechodzi - bez zgłoszenia i bez chwytu, dalej podejście z innej strony
    albo porzucenie; przechodzi - zgłoszenie z nowym P_rep. Postój w pozycji przed chwytem to
    jeden punkt widzenia, więc jedno pominięcie (P_d ≈ 0,88) zbija P_rep z ~0,99 do ~0,90:
    odrzucane są głównie zgłoszenia blisko progu (zwłaszcza trzecie, τ = 0,95), nie każdy brak
    detekcji z bliska. Wartość P_rep z bliska jest logowana (zdarzenie „weryfikacja: z bliska”)
    - do sprawdzenia w Gazebo, czy prawdziwe próbki nie są odrzucane (szczęki w kadrze).
17. **Margines kadru kamery przedniej (2026-10-25, decyzja autora).** Komórka jest w V(q) tylko,
    gdy punkt P_i leży co najmniej `frame_margin_deg` = 5° od krawędzi kadru (test w płaszczyźnie
    obrazu na kadrze pomniejszonym o margines). W Gazebo na końcu dojazdu (ρ ≈ 0,8 m, β 25-30°)
    prawdziwa próbka była 0,2-2,6° nad dolną krawędzią kadru - detektor (4 punkty na osi walca
    20 cm) jej nie widział, a model liczył obserwację negatywną z P_d = 0,99 (kalibracja M6
    0,3-1,0 m z 4 par, bez zależności od α); bramka różnorodności z bliska co ~20 cm daje nowy
    kubełek odległości, więc P_k spadło z 0,93 do zera w 4 s, przed skanem nadgarstkiem.
    Nadgarstek bez marginesu (`frame_margin_deg` = 0: ramię celuje w ŝ, kadr z 0,3 m ±0,13 m,
    środek komórki do 0,1 m od próbki). Atrapy, detektor ślepy w pasie 5° przy krawędzi,
    6 epizodów bez fałszywych alarmów: porzucenia prawdziwych próbek 9 → 5. Do zrobienia przy M6:
    P_d z bliska z większej liczby póz i z zależnością od położenia w kadrze.
18. **Cel chwytu z mediany pomiarów z bliska (2026-10-25, decyzja autora).** W Gazebo dwa kolejne
    pomiary z pozycji przed chwytem różniły się o 2,5-4,4 cm (Δz ±1,6 cm przy leżącej próbce -
    szum, nie ruch próbki); jedna korekta do ostatniego pomiaru, a chwyt do pozycji z tej korekty
    - palce chybiły (tolerancja w poprzek osi ±1,75 cm). Teraz cel to mediana wszystkich pomiarów
    z bliska danego zgłoszenia, najwyżej 2 korekty pozycji przed chwytem (> 1 cm), a chwyt liczy
    pionowy odcinek do końcowego celu, także gdy korekt już nie ma.
19. **Poza dojazdu poza skupiskiem kandydata (2026-10-25, decyzja autora).** Poza dojazdu
    odpada, gdy pod łazikiem (koło o promieniu travel.robot_radius = 0,55 m) jest więcej niż
    `max_mass_under_rover` = 10% P_k skupiska (P co najmniej jednej próbki w komórkach pod
    łazikiem). W Gazebo estymata ŝ z daleka myliła się o 0,66 m i poza 0,78 m od ŝ stawiała
    łazik 0,28 m od prawdziwej próbki - kamera przednia jej nie widziała (obserwacja negatywna
    zabiła kandydata), a próbka mogła zostać potrącona. Gdy żadna poza nie spełnia warunku -
    ta z najmniejszą masą pod łazikiem. Ponowne podejście też używa skupiska z beliefu.
20. **R(q) poniżej tabeli IK i pusty pierścień (2026-10-25, decyzja autora).** r_max(dz) dla dz
    poniżej tabeli (-6 cm) przedłużone liniowo (~2 cm zasięgu na 1 cm wysokości; przy dz ≈ -7 cm
    pierścień z zapasem jest pusty, przy -9 cm R(q) pusty), a nie wartość brzegowa 0,62 m.
    Z ramieniem pozy dojazdu, z których pierścień R(q) z zapasem jest pusty, odpadają (gdy
    wszystkie - najszerszy pierścień). W Gazebo z 3 z 8 kierunków próbka leżała 7-14 cm niżej
    niż łazik, a korekta dojazdu nie miała gdzie się zatrzymać (dojeżdżała do r_min).
21. **Minimalne P_k nowego dojazdu (2026-10-25, decyzja autora).** Strategia wybiera dojazd tylko
    do kandydatów z P_k ≥ `p_approach_min` = 0,1 (bieżący cel dojazdu zawsze); słabsze skupiska
    zostają w beliefie do obserwacji z daleka. W Gazebo dojazdy do P_k 0,04-0,09 kończyły się
    porzuceniem po ~3 s (P_k < p_abandon = 0,05), także dwukrotnie do tego samego miejsca.
22. **Garby w przejezdności (2026-10-25, decyzja autora).** Komórka nieprzejezdna także, gdy
    najwyższy garb ponad lokalną średnią (bez trendu zbocza) w promieniu łazika przekracza
    `bump_max` = 0,15 m (prześwit pod platformą ~0,20 m = wheel_radius + suspension_height -
    suspension_z). W Gazebo łazik zawisł na skale (garb 43 cm), choć nachylenie po wygładzeniu
    i chropowatość (odch. std.) były w normie. Przejezdne erc2025: 53% → 44% komórek.
23. **Utknięcie: wykrycie i odzyskanie (2026-10-25, decyzja autora).** Gdy w oknie 30 s od
    wysłania celu droga estymowanej pozy ≥ 3 m, a przesunięcie netto ≤ 0,5 m (koła buksują:
    odometria liczy ruch, GNSS ściąga pozę), cel jest przerywany, łazik cofa się prosto o 0,8 m
    (na czas - poza niewiarygodna), komórki w promieniu 0,8 m wypadają z grafu przejezdności
    i z póz obserwacji, a cele czekają na koniec cofania. W Gazebo łazik zawieszony na skale
    przez 26 min „jechał” w miejscu (odometr misji ~200 m przy 0 m w Gazebo), a 15 kolejnych
    dojazdów przerwał dopiero watchdog czasu. Okno 20 s / 2 m dało w Gazebo 2 fałszywe alarmy
    na 3 (łazik jechał slalomem: 1,3-1,4 m netto w Gazebo przy 0,3-0,5 m w estymowanej pozie).
24. **Chwyt: powrót do celu pozycji przed chwytem (2026-10-25).** Pionowy odcinek liczony do
    końcowego celu (pkt 18, oś szczęk z pozycji przed chwytem); gdy IK go nie ma (cel przy
    granicy zasięgu), do celu, nad którym ramię już stoi - zamiast rezygnacji z chwytu.
25. **Wysokość osi próbki do chwytu z mapy wysokości (2026-10-25).** Pozycja przed chwytem
    i chwyt dostają `SamplePose.ground_dz` (grunt pod próbką względem gruntu pod łazikiem,
    mapa wysokości); adapter przycina wysokość osi w base_link do [ground_dz, ground_dz + 4,5 cm]
    (promień próbki 1,5 cm, zgodność mapy wysokości p95 4 cm). W Gazebo pomiar z bliska podniósł
    z o 8,1 cm względem skanu, a chwyt przy zgodnych pomiarach xy zamknął się na pusto. Do
    potwierdzenia w kolejnym epizodzie (log: „oś próbki ... przycięta”).
26. **Cele osiągalne i w mapie kosztów Nav2 (2026-10-25).** Strategia nie wybiera akcji
    z T = ∞ albo S ≤ 0; trasy z odciętej wysepki grafu (składowa < 5% największej, np. po
    wyłączeniu miejsca utknięcia) liczone od najbliższej komórki dużej składowej; globalna mapa
    kosztów Nav2 90 × 90 m zamiast 60 × 60 (okno toczone; teren erc2025 43 × 30 m). W Gazebo
    z rogu mapy cele > 30 m były odrzucane jako poza mapą, a odcięty łazik przez 1050 s 5 razy
    na sekundę wybierał obserwację z T = ∞ (5323 zdarzenia „akcja”).
27. **Zgłoszenie z nadgarstkiem wymaga detekcji nadgarstka (2026-10-25, decyzja autora).** Gdy
    skan nie dał żadnej detekcji nadgarstka przy ŝ, reguła zgłoszenia nie przechodzi (ponowne
    podejście / porzucenie), nawet przy P_rep ≥ τ_n. W Gazebo kamień wyglądający jak próbka
    z kilku stron dał P_k = 1,00, skan nadgarstka 0 detekcji bez obniżenia P_rep (komórka 21 cm
    nad łazikiem poza V nadgarstka) i fałszywe drugie zgłoszenie (P_rep = 0,90 ≥ 0,85).
28. **Lokalizacja z Gazebo w serii głównej (2026-10-25, decyzja autora; PDR D3).**
    `sim_episode.launch.py` domyślnie `localization:=ground_truth`: map -> odom z prawdziwej pozy
    Gazebo, odom -> base_link z EKF. Z korekcją GNSS (CEP 1 m) błąd bezwzględny pozy wynosił
    średnio 10,2 cm (max 21 cm, stałe przesunięcie ok. (+4,6; -5,1) cm), a błędy zgłoszeń 6-11 cm
    przy ε = 15 cm - wynik zależał bardziej od lokalizacji niż od strategii. Wariant GNSS:
    `localization:=gnss`. W trybie ground_truth base_link ma w map bezwzględne z - wysokości
    do ramienia są liczone względem gruntu pod łazikiem (pkt 25 i poprawka skanu), więc działają
    w obu trybach.
29. **Klatki nadgarstka z pozycji przed chwytem bez wpływu na belief (2026-10-25, decyzja autora;
    wycofanie pkt 16).** Z ~0,3 m kadr nadgarstka to ±0,13 m, a P_d z tej odległości nie było
    kalibrowane (M6: poza skanu 0,5 m); pomiar z bliska często nie widział prawdziwej próbki.
    W Gazebo (lokalizacja z Gazebo) puste klatki z pozycji przed chwytem zbiły P_rep prawdziwej
    próbki z 1,00 do 0,20, kolejne skany (po 5 detekcji) podniosły je tylko do 0,52 i próbka
    została porzucona. Klatki z pozycji przed chwytem służą teraz tylko do pomiaru pozycji
    (mediana, pkt 18); warunek „P_rep ≥ τ_n (z bliska)” usunięty z drzewa - przed fałszywymi
    alarmami chroni wymóg detekcji nadgarstka przy skanie (pkt 27).

30. **Odstęp od krawędzi mapy wysokości (2026-10-26, decyzja autora).** Komórki bliżej niż
    `edge_margin` = 1,5 m od krawędzi ważnej części mapy wysokości (także brzegu siatki) są
    nieprzejezdne. W Gazebo (lokalizacja z Gazebo) łazik jechał 0,6 m/s do pozy obserwacji przy
    południowej krawędzi, wyjechał 0,25 m za ostatnią przejezdną komórkę (odstęp był tylko
    promieniem łazika, 0,55 m; tolerancja celu Nav2 0,4 m) i spadł z uskoku poza mapą
    (pochylenie 67-89°) - przeszkód ujemnych chmura Nav2 nie oznacza. Przejezdne erc2025:
    44% → 37%.

31. **Mapa przejezdności w stylu kalman_nav2 (2026-10-26, decyzja autora).** Binarna maska terenu
    (`config/maps/erc2025_traversability.{png,yaml}`, format nav2_map_server jak
    kalman_robot/kalman_nav2/maps: biały - przejezdny) z kryteriów TravelModel bez promienia
    łazika (nachylenie ≤ 20°, chropowatość, garby, odstęp 1,5 m od krawędzi mapy wysokości)
    i bez wzniesień - terenu ponad 0,25 m wyżej niż mediana w kwadracie 6 m (domknięte kołem
    0,5 m, dziury zalane): kopiec o łagodnych zboczach (6° po wygładzeniu) przechodził kryteria
    nachylenia, a spawner kładł na nim próbkę (seed 1, (15,51; −5,41), płaski wierzchołek 0,5 m
    nad otoczeniem; uwaga autora). Wygładzona (domknięcie i otwarcie kołem 0,3 m, dziury < 2 m²
    zalane), jedna spójna strefa z punktem startu (0, 0); 0,05 m/px, 496 m². Strefy wyłączone
    ręcznie (`config/maps/erc2025_traversability_exclude.yaml`, wielokąty w ramce map, opcja
    `--exclude`): pas na północ od długiego grzbietu (lewy górny róg mapy, x od −3,5 do 27,5 m),
    dostępny tylko przez wąskie przesmyki (uwaga autora); reguła „tylko przez przejście węższe
    niż 2 m” odcinała go jedynie częściowo, bo w masce przesmyki mają 1,5-2,5 m. Generator: `ros2 run ig_search
    traversability_map`. Mapy kalman_robot nie dało się użyć wprost: maska erc2025 dopasowana
    (obrót, odbicie, przesunięcie) do przejezdności naszego świata trafia najwyżej w 83%
    i równie dobrze w wielu miejscach - to inna arena (teren na miejscu zawodów, nie świat
    z symulatora Unity). Maska jest używana w trzech miejscach:
    - TravelModel: komórka przejezdna tylko na masce (odstęp `robot_radius` liczony dalej od
      jej brzegu); przejezdne erc2025: 37% → 27%; klucz cache póz obserwacji zawiera maskę;
    - prior p_i⁰: jednorodny tylko na masce (powierzchnia nośnika 1064 → 496 m²), bo próbki są
      tam rozmieszczane - rozkład spawnu i prior są zgodne;
    - Nav2 (`navigation.launch.py map:=...`): map_server publikuje /map, globalna mapa kosztów
      ma warstwę statyczną przed przeszkodami z kamery - planer omija zbocza, uskoki i krawędź
      mapy, których chmura przeszkód nie oznacza (przeszkody ujemne).
    Spawner (`probe_supervisor`, `traversability_map`, `map_clearance` = 0,5 m) losuje próbki
    w prostokącie area_* tylko na masce, co najmniej 0,5 m od jej brzegu.
32. **Próbka przyspawana do chwytaka, zebrana - usunięta (2026-10-26, decyzja autora; tylko
    symulacja).** W epizodach 9 i 10 chwycona próbka (trzymana tylko tarciem w DART) wypadała
    w drodze do pozycji transportowej. Teraz po zamknięciu szczęk na próbce (rozstaw w HOLD_GAP)
    szczęki są przytrzymywane w osiągniętym rozwarciu, a ig_search publikuje ~/sample_grasped
    (punkt między szczękami). `sample_store` sprawdza, że oś którejś próbki jest bliżej niż
    4 cm od tego punktu (szczęki naprawdę ją objęły), zastępuje ją kopią `grasped_probe` w tej
    samej pozie i DetachableJoint w modelu łazika (6DOF.urdf.xacro) spawa kopię
    z tcp_connector; podniesienie czeka na potwierdzenie (~/sample_attached, najwyżej 5 s).
    Po dojściu do pozycji transportowej (~/sample_stored) kopia jest odłączana i usuwana
    z Gazebo; przy otwarciu chwytaka z próbką (~/sample_released) - odłączana i zastępowana
    próbką o pierwotnej nazwie, która spada. Kopia pod stałą nazwą, bo DetachableJoint ma
    model dziecka w konfiguracji i dołącza go, gdy tylko ten istnieje. `probe_watch` liczy
    próbkę jako zebraną od chwili podmiany (wcześniej - od usunięcia w pozycji transportowej);
    upuszczona wraca do puli.
33. **Strategia z horyzontem `ig_horizon` (2026-10-26, decyzja autora; M10).** Przeszukiwanie
    wiązkowe po sekwencjach D = 3 obserwacji (`horizon_depth`), wiązka B = 10
    (`beam_width`), z każdego węzła N = 20 rozszerzeń najlepszych zachłannie
    (`beam_candidates`). Między obserwacjami belief jak przy braku detekcji (§3.6).
    Kryterium - zgłoszenia na sekundę dla całej sekwencji, **bez kary za czas f(t)** (f ≡ 1;
    f(t) później, osobno w ablacji): J(σ) = Σ_k g_k / Σ_k (T_k + t_follow,k), gdzie T_k to
    dojazd od poprzedniej pozy, a t_follow,k - czas od obserwacji do zgłoszenia jak
    w ig_greedy, liczony przy każdej obserwacji (rozłożony na sekwencję zawyżał J ok. 2× względem
    oceny dojazdu do kandydata). Dla D = 1 J = S_obs z ig_greedy. Dojazd do kandydata tylko jako
    pierwsza akcja, porównywany z J. Wykonywana jest pierwsza akcja najlepszej sekwencji,
    a `value_of` (histereza) ocenia najlepszą sekwencję zaczynającą się od bieżącej pozy.
    Czasy jazdy z póz wiązki z Dijkstry, z cache po komórce startowej i grafie. Planowanie na
    erc2025 (2808 póz): ok. 0,4 s (ig_greedy 0,03 s).
34. **Ramię nad próbką przy krótkim podjeździe w zasięg (2026-10-26, uwaga autora).** Gdy po
    skanie ŝ jest poza R(q), a podjazd potrzebny, żeby weszła w R(q), jest krótszy niż
    `ARM_OUT_REACH_MOVE` = 0,5 m (wzdłuż promienia od podstawy ramienia, ŝ w kącie R(q)),
    ramię zostaje w pozie skanu (kamera ok. 0,5 m nad próbką) i po podjeździe idzie od razu do
    pozycji przed chwytem - bez złożenia do pozycji transportowej (zamknięcie chwytaka)
    i ponownego otwarcia. W Gazebo podjazd w zasięg miał ok. 0,15 m (ruch bez Nav2), a od
    końca skanu do startu pozycji przed chwytem (złożenie, podjazd) mijało ok. 34 s ściennych
    (seed 1, ig_horizon, przebieg z GUI). Dłuższy
    podjazd, ponowne podejście i porzucenie - nadal ze złożonym ramieniem.
35. **Próbki tylko tam, gdzie nie znikają w podłożu (2026-10-26, uwaga autora; tylko
    symulacja).** Gazebo liczy kontakt z uproszczonym meshem kolizyjnym terenu (7024 trójkąty),
    a kamery widzą mesh wizualny. Tam, gdzie wizualny leży wyżej, próbka ⌀3 cm zapada się
    pod widoczne podłoże. Seed 1, `probe_object_1` w (3,52; 5,34): mesh wizualny 0,103 m,
    kolizyjny 0,070 m, środek próbki 0,085 m, czyli cała pod widoczną powierzchnią; nie
    została znaleziona. Mapa miejsc na próbki (`config/maps/erc2025_spawn.{png,yaml}`, z opcją
    `--collision` generatora; mapa wysokości z mesha kolizyjnego:
    `erc2025_collision_heightmap`) to mapa przejezdności bez komórek, w których różnica
    wizualny − kolizyjny przekracza 1,5 cm w kwadracie 0,5 m. Okno, bo próbka po upadku
    stacza się ~10 cm i ma 0,2 m długości. Zostaje 268 z 496 m². Spawner sprawdza ją
    w samym punkcie, a odstęp 0,5 m tylko od brzegu mapy przejezdności; 0,5 m od brzegu
    poszarpanej mapy miejsc zostawiałoby 5% prostokąta losowania. Prior p_i⁰ **nie** uwzględnia
    mapy miejsc (decyzja autora): to artefakt symulacji (niezgodność meshy), a nie wiedza
    o terenie - prior pozostaje jednorodny na mapie przejezdności (pkt 31). Jazda i Nav2 bez
    zmian: dla kół różnice ≤ 5 cm nie mają znaczenia.
36. **Bliski cel obserwacji bez Nav2 (2026-10-26).** Cel bez `precise` bliżej niż 0,6 m od
    łazika to sam obrót w miejscu do kursu celu (tolerancja 0,25 rad, do 0,6 rad/s), bez
    podjazdu - z tak bliska widok jest praktycznie ten sam. Pierwsza wersja z podjazdem
    (tolerancja 0,3 m) zawodziła: przy 0,5 m regulator przełączał jazdę przód/tył i kręcił się
    do limitu 25 s. Komendy prędkości regulatora mają zerowy znacznik czasu (sterownik napędu
    wstawia bieżący czas): znacznik z zegara węzła bywał ~1 s starszy od zegara Gazebo
    i sterownik odrzucał komendy jako przeterminowane. Regulated Pure Pursuit nie radził sobie ze ścieżką krótszą niż
    odległość wyprzedzenia: w Gazebo (frontier, seed 1) łazik 0,35 m od pozy o innym kursie
    drgał w miejscu ~70 s czasu misji (Nav2 4× „Failed to make progress”, obrót 90°),
    aż przerwało to wykrycie utknięcia. Zmiana tylko w infrastrukturze, model bez zmian.
37. **Wysokość próbki z nadgarstka: korekta wzdłuż promienia kamery (2026-10-26, decyzja
    autora).** Detektor (pose3d) szacuje odległość punktów próbki siecią z jednego obrazu
    (kamera nadgarstka w symulacji to dwie zwykłe kamery bez głębi; detektor tylko na lewej).
    Z bliska myli się o 8-14 cm, a pozycja przesuwa się wzdłuż promienia od kamery.
    Wcześniej wysokość osi spoza pasa nad gruntem była tylko przycinana, a x, y zostawały
    przesunięte razem z błędem; chwyty chybiały. Teraz `SamplePose` niesie kierunek
    promienia kamera→próbka (`ray`, `local_ray`) i punkt jest przesuwany wzdłuż niego do
    pasa nad gruntem (`domain/ray_height.slide_to_band`). Bez promienia, przy promieniu
    prawie poziomym (|r_z| < 0,2) albo korekcie > 0,3 m - samo przycięcie z, jak dotąd.
    Stereo z obu kamer nadgarstka - ewentualnie później.
    **Wycofane (2026-10-26, po pomiarze):** założenie było błędne. x, y z nadgarstka są trafne
    (zgłoszenia z niepoprawionych x, y: błąd 0,2-2,4 cm względem Gazebo), a błąd siedzi prawie
    wyłącznie w z (do −15 cm). Przesunięcie wzdłuż promienia psuło x, y o 6-15 cm i przy każdej
    korekcie w dół o > 8 cm IK nie znajdowało pozycji przed chwytem (4/4). Powrót do samego
    przycięcia z do pasa nad gruntem; `domain/ray_height.py` usunięty.
38. **Odstęp póz obserwacji od terenu nieprzejezdnego - sprawdzony, wyłączony (2026-10-26).**
    Utknięcia przy lewym brzegu erc2025 (7 z 16) nie wynikały z terenu (nachylenie 0,5-2,4°,
    chropowatość 3-7 mm na meshu wizualnym i kolizyjnym) ani z klasyfikacji chmury przeszkód:
    syntetyczna głębia z mapy wysokości i nagrana `/front/obstacles` w miejscu utknięcia dały
    najbliższe przeszkody ≥ 2,4 m (prawdziwe zbocza i kamienie), przodem do zbocza - żadnych.
    Nav2 utykał na „collision ahead”, zawracając 2,0-2,6 m od nich. Parametr
    `views.edge_clearance` (pozy obserwacji i punkty meandra co najmniej tyle od terenu
    nieprzejezdnego) na erc2025: 1,0 m - 1888 póz, 97,7% priora widoczne z P_d ≥ 0,5,
    15/16 miejsc utknięć nadal z pozą w pobliżu; 1,5 m - 1016, 90,4%, 10/16; 2,0 m - 440, 81,1%,
    4/16; 2,5 m - 152, 47,0%, 2/16. Odstęp, który omija miejsca utknięć, odbiera większość
    póz (korytarze erc2025 mają 4-6 m szerokości) - domyślnie 0 (wyłączony), parametr zostaje
    do ablacji.
39. **Horyzont sprawdzania kolizji RPP - sprawdzony, bez zmiany (2026-10-26).** Argument
    `collision_horizon` (max_allowed_time_to_collision_up_to_carrot) w `navigation.launch.py`,
    domyślnie 1,5 s jak w nav2_params. Porównanie 0,9 s vs 1,5 s (meander, 1 m/s): seed 1 -
    utknięcia 3 vs 6, „collision ahead” 214 vs 294, czas misji 740 vs 861 s (pary przedzielone
    też korektą wysokości z pkt 37); seed 2 (ten sam kod) - utknięcia 4 vs 4, „collision
    ahead” 232 vs 298, czas 704 vs 616 s, droga 220 vs 186 m. Krótszy horyzont zmniejsza liczbę
    komunikatów o ~25%, ale nie liczbę utknięć ani czasu misji - zostaje 1,5 s (większy zapas
    bezpieczeństwa). Wszystkie cztery epizody 3/3 poprawnie.
40. **Nieudane nawigacje bez ruchu - odzyskanie jak po utknięciu (2026-10-27).** W serii
    porównawczej (ig_horizon, seed 4) łazik stanął 0,15 m od brzegu mapy przejezdności; z
    nadmuchaniem warstwy statycznej Nav2 każdy cel odrzucał od razu („Start occupied”), a misja
    wykluczała pozę i brała następną - 2615 celów w 590 s, łazik nie ruszył się, a wykrycie
    utknięcia (wymaga ruchu bez postępu) nie zadziałało. Teraz 3 nieudane nawigacje w 30 s
    z przesunięciem ≤ 0,3 m uruchamiają to samo odzyskanie co utknięcie (cofnięcie 0,8 m,
    miejsce wyłączone z grafu i z póz obserwacji). Epizody serii przed tą zmianą: ig_greedy
    seed 1-4, ig_horizon seed 1-4, meander i frontier seed 1-3 (problem wystąpił tylko raz).
41. **f(t) z regulaminu ERC (2026-10-27, decyzja autora).** Regulamin 7.3.4.4 a: ułamek
    zebranych punktów zależy od czasu ukończenia zadania - 100 % do 7:30, 90 % do 12:30,
    80 % do 17:30, 70 % do 22:30, potem 60 %. Funkcja schodkowa `MissionParams.score_factor`
    (`mission.score_steps`, `mission.score_factors`; zastępuje `full_score_time`); równo na
    progu liczy się wyższy stopień. Mnożnik dotyczy wszystkich punktów naraz i zależy od chwili
    końca misji, nie od chwil poszczególnych zgłoszeń. W analizie serii punktacja epizodu
    = (n_pop / M) · f(t_końca), epizod przerwany technicznie - bez punktacji. W planerze f(t)
    nie jest użyta: jest stała między progami, więc w kryterium Σ_k f(t_k)·g_k z §3.6 zmienia
    ranking sekwencji tylko wtedy, gdy horyzont (ok. 1-2 min) przekracza próg, a strategie
    oparte na beliefie kończą zwykle przed 7:30 (seria porównawcza: punktacja 0,98-1,00).
    Celem pozostaje minimalizacja t₃. Jedyna decyzja, którą f(t) mogłaby realnie zmienić,
    to wcześniejsze zakończenie misji (zgłoszenie mniej niż M próbek przed progiem) - opłaca
    się, gdy f(t')/f(t) < n/(n + 1). Przy tych stopniach tylko dla n = 2, gdy misja mogłaby
    skończyć się przed 7:30, a trzecia próbka byłaby znaleziona dopiero po 22:30
    (0,6 < 2/3). Niezaimplementowane.
42. **Uczciwe strategie odniesienia i symulator offline (2026-10-27, decyzja autora).**
    Obawa: przewaga `ig_` może wynikać z większego nakładu na jej dopracowanie. Zmiany:
    (a) meander i frontier po pokryciu terenu zaczynają kolejne przejście zamiast kończyć
    misję (`planner.coverage_passes`, 0 - bez limitu, 1 - dawne zachowanie; w serii
    porównawczej frontier na seedzie 1 kończył z 1/3). Frontier w przejściu k szuka komórek
    obejrzanych z < k różnych punktów widzenia (`Belief.view_counts`, kubełki bramki), meander
    przesuwa pasy o pół rozstawu i jedzie od końca. (b) Nowa strategia `chase`: dojazd do
    najbliższej surowej detekcji kamery przedniej, niezależnie od beliefu (detekcje < 0,5 m
    łączone, miejsca zgłoszone i porzucone pomijane, `Mission._note_raw`); cel z surowej
    detekcji nie jest porzucany przez belief do pierwszej weryfikacji na miejscu, potem jak
    każdy kandydat. Ta sama weryfikacja i reguła zgłoszenia; bez detekcji - eksploracja jak
    frontier. Oddziela zysk z beliefu od zysku z planowania. (c) Parametry strategii
    odniesienia (zakładka meandra, próg dojazdu, `frontier_min_gain`) strojone tak samo jak
    `ig_` - przeglądem na ziarnach strojenia. Do przeglądu - `ig_search_experiments
    offline_series`: misja z prawdziwym kodem domeny i aplikacji na mapie erc2025, próbki
    z kodem spawnera i tym samym ziarnem co w Gazebo, ale detektor testowy z modelem planera
    (jak `test/harness.py`), jazda po punktach pośrednich ze stałą prędkością, ramię jako
    atrapa (każdy ruch 8 s). **Wyłącznie do strojenia i wstępnej selekcji** - wyniki pracy
    nadal z Gazebo (PDR), z zamrożonymi konfiguracjami i na ziarnach nieużytych do strojenia.
43. **Realistyczny świat `erc2025_real` i próbki wbite pionowo (2026-10-27, decyzja autora;
    tylko symulacja).** Teren z erc_import ma prawdziwą rzeźbę i ortofoto toru (~1,3 cm/px),
    ale kamienie tylko namalowane; zdjęcia z zawodów pokazują płyty, głazy, odłamki, żwir,
    pochmurne światło i próbki wbite pionowo. `sirius_gazebo/tools/marsyard_dressing/
    generate_dressing.py` (stałe ziarno - jeden tor) generuje: model `erc2025_dressing` (952
    płyty w dwóch „ogródkach”, 84 głazy, 218 kamieni - fotoskany Poly Haven CC0 zmniejszone
    w Blenderze, 160 odłamków; kolizja tylko głazów ≥ 0,25 m - łazik osiadał podwoziem na
    18-cm kamieniu i buksował, a między płytami na brzegu ogródka więzł; prawdziwy po płytach
    przejeżdża, wystają 1-3 cm), teren `erc2025_marsyard_hd` (tekstura 3× gęstsza
    z fotograficznym detalem żwiru i piasku), świat `erc2025_real.sdf` (światło pochmurne),
    strefy głazów `config/maps/erc2025_dressing_exclude.yaml` i obrysy kamieni
    (`footprints.yaml`). Mapa przejezdności świata: `erc2025_real_traversability.yaml` (+
    `erc2025_real_spawn.yaml`; `traversability_map --exclude` przyjmuje teraz kilka plików).
    `sim_episode world_variant:=real` przełącza świat i obie mapy (także w węźle misji).
    Próbki wbite pionowo: `upright_fraction` spawnera - statyczne (ruchome ciało w terenie
    Gazebo wyrzuca), czubkiem w dół, zakopane 25-45 % długości, przechylone do 12°,
    wysokość z mapy wysokości mesha wizualnego, nie w kamienie i płyty; zebranie działa jak
    dotąd (podmiana na przyspawaną kopię). Dla `upright_fraction` = 0 ciąg losowy i pozycje
    próbek bez zmian. Rendery porównawcze: `render_views.py`.
    Uzupełnienie (2026-10-27, prośba autora o więcej kamieni i realistyczniejsze światło):
    112 głazów, 488 kamieni 7-30 cm (0,8 / m²), 4891 kamyków 3-8 cm (7 / m², proceduralne,
    tylko wizualne - żwir z bliska jako bryły), usypiska gęstsze o ~40 %; 80 głazów w mapie
    przejezdności (przejezdne 496 m², bez zmian). Światło: niebo proceduralne ogre2 z chmurami,
    rozproszone 0,36 zamiast 0,5, słońce mocniejsze (0,95) i niżej (dłuższe cienie). Globalne
    oświetlenie VCT systemu Sensors sprawdzone - bez widocznej różnicy (tekstura terenu to
    ortofoto z wypalonym światłem), +60 % czasu renderu - nieużyte.
44. **Utknięcie bez postępu (2026-10-27).** Kryterium z pkt 40 i wcześniejszych wymaga drogi
    z odometrii; z lokalizacją z Gazebo poza jest prawdziwa, więc przy buksujących kołach droga
    nie rośnie. W `erc2025_real` łazik osiadł na kamieniu i przez 16 min dostawał kolejne cele
    (watchdog przerywał każdy po ~100 s). Teraz także: cel aktywny ≥ `STALL_WINDOW` = 60 s
    i przesunięcie netto w tym oknie ≤ `STALL_NET` = 0,5 m - to samo odzyskanie (cofnięcie,
    miejsce wyłączone). Test w `test_mission.py`.
45. **Detektory v2 douczone na `erc2025_real` (2026-10-28, decyzja autora; tylko symulacja).**
    Modele v1 (`gz_front_v1`, `gz_wrist_v1`) uczone tylko na renderach starego świata.
    W `erc2025_real` przedni v1 systematycznie zawyżał odległość (×1,38; walidacja: recall
    0,67, 0,62 FP / obraz, mediana błędu głębokości 642 mm) - prawdopodobna przyczyna porażek
    pierwszych misji w tym świecie. Etykiety obu światów są spójne, więc to problem domeny
    obrazu (kamienie, żwir, światło). Dane z nowego świata: `oak_probe_vision/scripts/gazebo/
    generate_gz_dataset.py --world real` (łazik i kamera poza głazami, 60 % atrap to odłamki
    skał); douczenie od wag v1 na zbiorze obu światów (przedni 20 epok, nadgarstek 7).
    Walidacja v2 (erc2025_real / stary świat): przedni recall 0,83 / 0,85, 0,08 / 0,06 FP na
    obraz, mediana błędu głębokości 59 / 70 mm; nadgarstek 0,96 / 0,97, 0,04 / 0,03 FP,
    9 / 10 mm. Test `sim_probe_test` w `erc2025_real` (przedni v2 / v1): recall w kadrze
    0,58 / 0,40, FP na klatkę 0,02 / 1,19, mediana błędu środka 63 / 341 mm; v2 do 3 m
    wykrywa 95-100 %, 3-4 m 55 %, dalej nic. Znana słabość v2 w starym świecie: z bliska
    (< 0,6 m) przedni odwraca kierunek osi próbki, nadgarstek ma medianę błędu osi 32° zamiast
    12° (w `erc2025_real` 10°). Wagi v2 zastąpiły v1 w `probe_detector/models` (te same
    nazwy plików); v1 pozostają w `/mnt/dane/synth_data/runs/gz_*_v1`. Misje
    `erc2025_real` (3 ziarna, 30 min): v2 - 1 poprawne zgłoszenie, 172 m średnio; v1 - 0, 42 m
    (trzeci epizod v1 nie wystartował: od 12:44 Gazebo zawiesza się przy ładowaniu
    gz_ros2_control - awaria środowiska, nie nawigacji); przyczyny słabych misji (ziarno 3 v2
    przejechało tylko 17 m) do zbadania.
