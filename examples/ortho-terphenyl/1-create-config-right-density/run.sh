#!/bin/bash

uv run --script create-initial-config.py

densities=(0.2 0.22 0.24 0.27 0.30 0.33 0.36 0.40 0.44 0.48 0.53 0.58 0.64 0.70 0.77 0.85 0.93 1.02 1.12 1.2)

for density in "${densities[@]}"
do
    echo "Density" $density                                         # gibt den aktuellen Dichtewert im Terminal aus 
    sed "s/DENSITY/$density/" params-template.toml > params.toml    # params-template.toml ist eine Vorlage für die Simulationsparameter, darin: density = DENSITY, sed ersetzt DENSITY durch den aktuellen Wert von density, das Ergebnis wird dann in die Datei params.toml geschrieben, d.h. während die Simulation läuft ist params-template unverändert, während das skript params fortlaufend verändert wird, > bedeutet, die nachfolgende Datei wird neu bzw. überschrieben, HINWEIS: Ohne ein g am Ende des sed-Ausdrucks (s/DENSITY/$density/g...) wird pro Zeile nur das erste Vorkommen von DENSITY ersetzt, d.h. wenn DENSITY mehrfach in einer Zeile stünde, wäre das besser
    particlesmc params.toml                                         # führt das Programm particlesmc aus und übergibt dabei die erzeugte Konfigurationsdatei params.toml
    cp chains/1/lastframe.xyz inputframe.xyz                       # kopiert die Datei chains/1/lastframe.xyz nach inputframe.xyz, der letzte gespeicherte Zustand der gerade beendeten Simulation wird also als Eingabezustand für die nächste Simulation genutzt 
    echo "Done density: $density"                                  # Ausgabe im Terminal 
done     # beendet die for-Schleife 

echo "Final density: $density"   # Ausgabe der finalen Dichte im Terminal 

# Note: In Bash bleibt die Schleifenvariable nach der Schleife erhalten