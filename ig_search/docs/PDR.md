# PDR — ig_search, wariant B, weryfikacja w Gazebo

Zatwierdzony 2026-10-19. Model: `docs/DESIGN.md` (z poprawkami w §6 tamtego dokumentu).

## Decyzje

| # | Decyzja | Wybór |
|---|---------|-------|
| — | Środowisko badań | wyłącznie Gazebo Harmonic (świat `erc2025`), bez osobnego symulatora; przy braku czasu - mniej przebiegów |
| D1 | Detektor | istniejący `probe_detector` (model pose3d), wejście `Detection3DArray` - planer jest agnostyczny |
| D2 | Weryfikacja z bliska | kamera front + **OAK-D SR na nadgarstku**: skan weryfikujący nad ŝ i ustawienie chwytaka w pozycji przed chwytem (`manipulator:=6DOF`, MoveIt) |
| D3 | Lokalizacja | **zmiana 2026-10-25: prawdziwa poza z Gazebo (`localization:=ground_truth`) w serii głównej** - korekcja GNSS (CEP 1 m) dawała średnio 10 cm błędu bezwzględnego (max 21 cm) przy tolerancji zgłoszenia 15 cm, co przesłaniało ocenę strategii; EKF + korekcja GNSS (`localization:=gnss`) jako wariant (wcześniej, 2026-10-20: GNSS w serii głównej) |
| D4 | Informacja o fałszywym zgłoszeniu | brak (system traktuje zgłoszenie jako odnalezienie) |
| D5 | Reset epizodu | pełny restart stosu na epizod |
| D6 | Wiadomości | nowy pakiet `ig_search_interfaces` (`SampleReport`, `MissionStatus`) |

## Architektura

```
ig_search/
  domain/            bez ROS (numpy/scipy)
    params.py          SearchParams (YAML / parametry ROS z kropką)
    terrain.py         Grid, HeightMap
    mesh_heightmap.py  OBJ -> mapa wysokości (rasteryzacja)
    visibility.py      V(q) 2,5D, poza kamery z q i h
    detector_model.py  P_d(r, α), σ(r), λ_fa (parametrycznie / tablica z kalibracji)
    belief.py          log-szanse + explaining away + bramka różnorodności
    travel.py          graf przejezdności, czas z nachyleniem
    views.py           kandydackie pozy, macierz V (cache .npz)
    candidates.py      skupiska, P_k, ŝ_k, pozy dojazdu
    report_rule.py     P_rep, τ_n, R(q)
    strategies/        ig_greedy, meander, frontier (+ ig_horizon w M10)
  application/
    ports.py           Navigator, Manipulator, Reporter, Frame, Report
    world.py           SearchWorld (modele statyczne)
    mission_tree.py    drzewo zachowań misji (py_trees): kandydat -> zgłoszenie, przeszukiwanie
    mission.py         kontekst misji (belief, kandydat, zgłoszenia), fazy SEARCH..TIMEOUT
  infrastructure/    węzeł ROS 2, Nav2, MoveIt, publikacja (M5)
  presentation/      CLI mesh_to_heightmap, launch
ig_search_interfaces/  SampleReport.msg, MissionStatus.msg
ig_search_experiments/ sędzia, epizody, analiza (M7-M9)
```

## Kamienie milowe

| # | Zakres | Stan |
|---|--------|------|
| M0 | pakiety, notacja, parametry, testy | ✅ |
| M1 | mapa wysokości z mesha, widoczność 2,5D, prekomputacja V | ✅ (erc2025: 4456 póz, 17 s, cache 0,1 s) |
| M2 | czas ruchu z nachyleniem | ✅ offline; porównanie z Nav2 (±25%) - w M5 |
| M3 | belief + model detektora + bramka | ✅ |
| M4 | kandydaci, reguła zgłoszenia, misja, ig_greedy, meander, frontier | ✅ offline (atrapy portów) |
| M5 | węzeł ROS 2 + Nav2 + MoveIt, pierwszy epizod w Gazebo | ✅ pełne misje 3/3 we wszystkich strategiach (2026-10-26); spaw próbki w chwytaku, despawn po zebraniu (DESIGN §6.32) |
| M6 | kalibracja P_d(r, α), σ(r), λ_fa w Gazebo | ✅ tablice front/nadgarstek w `config/calib/` (DESIGN §6.13) |
| M7 | sędzia, automatyczne epizody, pomiar RTF | ✅ sędzia `probe_watch` (summary/status/robot/probes), seria `run_series` (wznawianie, przerwanie, ponowienie przy zawieszeniu Nav2); RTF 0,3 → 0,6 po ograniczeniu /clock do 100 Hz |
| M8 | strategie odniesienia w Gazebo | ✅ seria 2026-10-27, ziarna 1-5: t₃ mediana (KM) ig_greedy 346 s, ig_horizon 277 s, meander 699 s, frontier 725 s (4/5 kompletnych - po pokryciu terenu kończy); 0 fałszywych zgłoszeń |
| M9 | miary, Kaplan-Meier, wykresy | ✅ `analyze_series` (η, π, t₁-t₃ KM z cenzurowaniem, droga, błędy, entropia, pary ziaren) - wyniki serii M8/M10 |
| M10 | horyzont (beam search, f(t)) | ⏳ ablacja D (ziarna 1-5, wszystkie 3/3): t₃ mediana D=1 346 s, D=2 254 s, D=3 277 s, D=5 237 s, średnie 303 / 274 / 309 / 287 s - przewaga horyzontu niewielka i zmienna między ziarnami; f(t) schodkowa z regulaminu ERC (DESIGN §6 pkt 41) - punktacja śr.: D=1 1,00, D=2 1,00, D=3 0,98, D=5 1,00, meander 0,88, frontier 0,75 |

## Ryzyka

R1 RTF ≪ 1 · R2 GPU w symulacji vs CPU na łaziku (`detector_rate` w czasie symulacji) ·
R3 reset epizodu · R4 niedeterminizm Gazebo/Nav2 (pary seedów) · R5 model czasu ruchu vs Nav2 ·
R6 staczanie się próbek (sędzia bierze pozycje z Gazebo po ustaleniu) · R7 pamięć V.

## Wnioski z uruchomień w Gazebo (M5, 2026-10-19/20)

| Problem | Skutek | Rozwiązanie |
|---------|--------|-------------|
| Geometria: kamera przednia 0,392 m nad gruntem, nie widzi gruntu bliżej ~0,84 m; chwyt pionowy 0,60-0,74 m od podstawy ramienia na płaskim (zależnie od wysokości próbki), ±75° | weryfikacja z bliska możliwa tylko nadgarstkiem | dojazd do wejścia ŝ w R(q) (DESIGN §6.10-6.12); bez nadgarstka: weryfikacja z 1,15 m + podjazd |
| Tolerancja celu Nav2 0,4 m / 0,5 rad | poza obok łazika „osiągnięta” bez ruchu (zakleszczenie); dojazd poza pasem chwytu | wstrzymanie osiągniętej pozy na 60 s; dokładny dojazd (regulator na /cmd_vel), przejęcie od Nav2 bez postępu < 1 m |
| Nav2 krąży tuż za tolerancją / łazik utyka bez porażki | misja stoi setki sekund | watchdog: 3·T + 60 s |
| `probe_detector` z target_frame map przy spóźnionym TF porzuca klatki | 0 detekcji w misji | detektory w ramce optycznej, ig_search buforuje klatki do dostępności TF |
| Korekcja GNSS: błąd ~1,8 m, 37% CPU | belief w złym miejscu | D3: `localization:=ground_truth` (ig_search_experiments) |
| Timery adapterów tworzone w węźle lifecycle z czasem symulacji nie odpalały (0 wywołań) | dokładny dojazd i ramię martwe | adaptery bez timerów, `tick()` z timera węzła |
| /clock ~400 Hz (każdy krok fizyki), każdy węzeł rclpy z czasem symulacji 25-60% rdzenia | RTF ~0,3 | `clock_rate:=100` (topic_tools throttle, sirius_gazebo), spawner bez mostów pozycji w seriach: RTF ~0,6 |
| Menedżer cyklu życia Nav2 czasem zawisa na konfiguracji (`smoother_server`) | epizod bez nawigacji | `run_series` wykrywa (Nav2 nieaktywny po 300 s) i powtarza epizod |
| Nav2 (RPP) „collision ahead” 2-2,6 m od zboczy i kamieni | 3-6 utknięć na epizod meandra, każde ~30-40 s | wykrycie utknięcia i cofnięcie (DESIGN §6.23); odstęp póz i krótszy horyzont kolizji sprawdzone bez wyraźnej poprawy (§6.38-39) |
| Misja startowała przed Nav2 | wykluczanie póz | start po gotowości navigate_to_pose i move_group |
| RTF ≈ 0,32 (0,42 bez GNSS), CPU wysycone przez Nav2/MoveIt | 10 min misji ≈ 25-30 min | propozycje przyspieszenia czekają na decyzję autora |
| Model detektora nieskalibrowany (P_d do 6 m) | puste klatki „czyszczą” obszary, których detektor realnie nie widzi - próbki tracone | M6: tablice z Gazebo (DESIGN §6.13) |

## Wydajność symulacji (RTF), 2026-10-20

Pomiar A/B na żywej misji (headless, wyłączanie kolejnych procesów): główny koszt to fizyka
jadącego łazika na meshu terenu; procesy ROS mają mały udział (Nav2, MoveIt, EKF ~0; detektory
~0,06; subskrypcja pozycji wszystkich modeli z Gazebo ~0,09). Wdrożone: Nav2 bez route_server,
docking i waypoint_follower (`ig_search/launch/navigation.launch.py`), `servo:=false`,
kamery w symulacji 10 Hz - bez mierzalnej zmiany RTF.

Krok fizyki, ta sama misja (seed 1, pełny stos, 4 przebiegi naprzemiennie):

| Krok | RTF (średnia ± odch. std, dwa przebiegi) |
|------|------------------------------------------|
| 1 ms | 0,41 ± 0,11 · 0,42 ± 0,12 |
| 2 ms | 0,74 ± 0,20 · 0,76 ± 0,22 |

Zachowanie jazdy (sama Gazebo, ta sama sekwencja /cmd_vel): rozrzut trajektorii między
powtórzeniami przy 1 ms jest duży (punkty końcowe do 4 m od siebie), max przechył/pochylenie
1 ms 15-32° / 26-40°, 2 ms 23-37° / 26-39° (w rozrzucie 1 ms), 4 ms 29° / 66° (niestabilnie,
odrzucone). Decyzja o kroku 2 ms - autor.
