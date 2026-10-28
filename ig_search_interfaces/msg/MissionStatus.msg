# Stan misji przeszukiwania, publikowany cyklicznie.
std_msgs/Header header

uint8 SEARCH = 0
uint8 APPROACH = 1
uint8 VERIFY = 2
uint8 REPORT = 3
uint8 DONE = 4
uint8 TIMEOUT = 5
uint8 state

string strategy               # nazwa strategii (np. ig_greedy, meander)
uint8 reports                 # liczba dotychczasowych zgłoszeń
float64 elapsed               # [s] czas od startu misji (czas symulacji)
float64 belief_mass           # suma p_i - oczekiwana liczba nieodnalezionych próbek
float64 belief_entropy        # suma entropii komórek [bit]
string detail                 # opis bieżącej akcji (do logów)
