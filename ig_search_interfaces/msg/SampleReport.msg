# Zgłoszenie odnalezienia próbki (ig_search -> sędzia / moduł chwytania).
std_msgs/Header header        # ramka map, czas zgłoszenia

uint8 index                   # numer zgłoszenia n = 1, 2, 3
geometry_msgs/Point estimate  # estymata położenia próbki ŝ
float64 probability           # P_rep w chwili zgłoszenia
float64 threshold             # próg τ_n, który został przekroczony
geometry_msgs/Pose robot_pose # poza łazika q przy zgłoszeniu

# Poza próbki z kamery na nadgarstku (oś z = oś próbki); ważna, gdy has_sample_pose.
bool has_sample_pose
geometry_msgs/Pose sample_pose
bool pregrasp_reached         # ramię ustawione w pozycji przed chwytem
bool grasped                  # próbka chwycona i podniesiona
