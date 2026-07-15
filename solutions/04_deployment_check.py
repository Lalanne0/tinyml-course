#!/usr/bin/env python3
# =============================================================================
#  EXERCICE 04 : Validation du deploiement (TinyEngine)  --  SOLUTION
# =============================================================================

import os
import sys
sys.path.insert(0, "/workspace/scripts")
from coursekit import banner, section, reflexion, next_step
from tinyengine_analysis import analyze_tflite, print_report

banner("04", "Validation du deploiement (TinyEngine) (SOLUTION)",
        "Verifier si un modele tient dans la memoire d'un MCU", "40 min")

# -----------------------------------------------------------------------------
# 1. Analyse de mcunet-vww1 sur STM32F746
# -----------------------------------------------------------------------------
section("1. Analyse de mcunet-vww1 sur STM32F746")

tflite_vww1 = os.path.expanduser("~/.mcunet/mcunet-vww1.tflite")
cible_mcu = "STM32F746"

print(f"Lancement de l'analyse pour {cible_mcu}...\n")
result_vww1 = analyze_tflite(tflite_vww1)
print_report(result_vww1, cible_mcu)
print("\nmcunet-vww1 passe sans probleme (pic SRAM ~160 KB).")

# -----------------------------------------------------------------------------
# 2. Exercice : analyser mcunet-vww2 sur STM32F746  (SOLUTION)
# -----------------------------------------------------------------------------
section("2. Analyse de mcunet-vww2 sur STM32F746")

tflite_vww2 = os.path.expanduser("~/.mcunet/mcunet-vww2.tflite")

print(f"Lancement de l'analyse pour mcunet-vww2 sur {cible_mcu}...\n")
result_vww2 = analyze_tflite(tflite_vww2)
print_report(result_vww2, cible_mcu)
print("\nDebordement memoire : le modele ne pourra pas etre flashe sur ce MCU.")

reflexion([
    "Pic SRAM inferieur avec le vrai TinyEngine -> il fait de la fusion "
    "d'operateurs (in-place updates) agressive : si la sortie d'une operation peut "
    "ecraser son entree sans alterer le calcul, le pic SRAM de la couche est "
    "divise par deux. L'analyseur TFLite standard ne voit pas cette optimisation.",
    "Avantage du code C statique (TinyEngine) -> pas d'allocation dynamique a "
    "l'execution, pas d'overhead (Flash + SRAM) de l'interpreteur, boucles "
    "deroulees et optimisees pour la cible a la compilation. Plus leger et bien "
    "plus rapide que TFLite Micro.",
])

next_step("exercises/05_what_if_it_doesnt_fit.py")
