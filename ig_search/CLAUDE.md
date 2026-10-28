# CLAUDE.md

Pakiet ROS 2 (Jazzy, ament_python) do przeszukiwania terenu w poszukiwaniu M = 3 próbek
(konkurencja ERC Probing, Mars Yard, łazik Sirius 2). Projekt jest częścią pracy dyplomowej.

## Najpierw przeczytaj
- `docs/PDR.md` — decyzje, architektura, kamienie milowe M0–M10 i ich stan.
- `docs/DESIGN.md` — model (teren 2,5D, detektor z fałszywymi alarmami, dojazd i zgłoszenie,
  horyzont, eksperymenty), notacja zgodna z pracą; **§6 — zmiany modelu wprowadzone przy implementacji**.

## Struktura (Clean Architecture)
- `ig_search/domain/` — teren, widoczność, belief, model detektora, ruch, pozy, kandydaci,
  reguła zgłoszenia, strategie. Bez ROS.
- `ig_search/application/` — porty (Navigator, Manipulator, Reporter), `SearchWorld`, drzewo zachowań misji
  (`mission_tree.py`, py_trees) i jego kontekst `Mission` (`mission.py`).
- `ig_search/infrastructure/` — węzeł ROS 2 i adaptery (Nav2, MoveIt) — M5.
- `ig_search/presentation/` — CLI (`mesh_to_heightmap`), launch.
- `config/maps/erc2025_heightmap.{yaml,npy}` — mapa wysokości z wizualnego mesha (0,1 m).
- `test/` — pytest bez ROS; `harness.py` — atrapy portów i prosty detektor testowy
  (tylko do testów logiki, nie do eksperymentów).
- Wiadomości: pakiet `ig_search_interfaces`.

## Komendy
- Testy (system ma zepsuty pytest-cov): `python3 -m pytest -p no:pytest_cov -q test/`
  (lintery ament wymagają `source /opt/ros/jazzy/setup.bash`).
- Budowa: `colcon build --packages-select ig_search_interfaces ig_search --symlink-install`
- Mapa wysokości: `ros2 run ig_search mesh_to_heightmap <teren.obj> <out.yaml> --res 0.1`
- Mapa przejezdności (Nav2, spawn próbek, TravelModel): `ros2 run ig_search traversability_map config/maps/erc2025_heightmap.yaml config/maps/erc2025_traversability.yaml --exclude config/maps/erc2025_traversability_exclude.yaml --collision config/maps/erc2025_collision_heightmap.yaml --spawn-out config/maps/erc2025_spawn.yaml` (z mapą miejsc na próbki)

## Zasady
- Logika bez importów ROS — musi działać i być testowalna offline. ROS tylko w infrastructure.
- Obliczenia wektorowo w numpy; prekomputacja tam, gdzie mapa jest znana (cache V w `.npz`).
- Notacja zgodna z `docs/DESIGN.md` §1 (belief `p`, log-szanse `ell`, poza `(x, y, psi)`).
- Eksperymenty wyłącznie w Gazebo (decyzja PDR); detektor testowy z `test/harness.py` nie służy do oceny strategii.
- Wszystko losowe przez jawny seed (`numpy.random.Generator`).
- Dokumentacja i komunikaty po polsku, identyfikatory po angielsku.
- Styl ament: importy w jednej grupie alfabetycznej po stdlib, docstringi wielowierszowe od drugiej linii, 99 kolumn.
- Przy niejasnościach modelu zapytaj autora zamiast zgadywać; zmiany modelu dopisuj do DESIGN §6.
