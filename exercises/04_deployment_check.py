#!/usr/bin/env python3
# =============================================================================
#  Deploiement TinyML avec MCUNet
#  EXERCICE 04 : Validation du deploiement avec TinyEngine
# =============================================================================
#
#  Objectif : Valider qu'un modele est reellement compatible avec un MCU cible
#             en utilisant une analyse statique de la memoire (a la TinyEngine).
#
#  Concepts cles :
#   1. TinyEngine       : moteur d'inference de MCUNet. Contrairement a TFLite
#      Micro (un interpreteur), c'est un GENERATEUR de code : il analyse le
#      graphe, planifie la memoire a l'avance (Ahead-Of-Time), et genere du C.
#   2. Memory Scheduling: reutiliser les memes blocs de SRAM pour differents
#      tenseurs a differents moments, pour minimiser le pic memoire.
#
#  Prerequis : comprendre les metriques SRAM/Flash, avoir un fichier .tflite.
# =============================================================================

import os
import sys
sys.path.insert(0, "/workspace/scripts")
from coursekit import banner, section, todo_stop, reflexion, next_step

# Le wrapper Python fourni simule l'analyse statique de TinyEngine.
from tinyengine_analysis import analyze_tflite, print_report

banner("04", "Validation du deploiement (TinyEngine)",
        "Verifier si un modele tient dans la memoire d'un MCU", "40 min")

# -----------------------------------------------------------------------------
# 1. Analyse statique d'un modele qui rentre
#    (Dans un projet reel on compilerait TinyEngine en C++. Ici on utilise le
#    wrapper scripts/tinyengine_analysis.py qui simule cette analyse.)
# -----------------------------------------------------------------------------
section("1. Analyse de mcunet-vww1 sur STM32F746")

tflite_vww1 = os.path.expanduser("~/.mcunet/mcunet-vww1.tflite")
cible_mcu = "STM32F746"

print(f"Lancement de l'analyse pour {cible_mcu}...\n")
result_vww1 = analyze_tflite(tflite_vww1)
print_report(result_vww1, cible_mcu)
print("\nmcunet-vww1 passe sans probleme ! Le pic SRAM est coherent avec le "
      "Model Zoo (~160 KB).")

# -----------------------------------------------------------------------------
# 2. Exercice pratique : tester les limites
#    Que se passe-t-il si on essaie de deployer mcunet-vww2 sur le meme MCU ?
#    L'analyser avec la meme methode.
# -----------------------------------------------------------------------------
section("2. Exercice : analyser mcunet-vww2 sur STM32F746")

# ===== TODO ==================================================================
#  1. Recuperer le chemin vers mcunet-vww2.tflite dans le cache (~/.mcunet/).
#  2. L'analyser avec analyze_tflite(...) et stocker le resultat dans result_vww2.
#  3. Afficher le rapport avec print_report(result_vww2, cible_mcu).
result_vww2 = None  # <-- remplacer None par votre appel a analyze_tflite(...)
# =============================================================================

if result_vww2 is None:
    todo_stop("exercises/04_deployment_check.py", "section 2 (result_vww2)")

print_report(result_vww2, cible_mcu)
print("\nOups ! Debordement memoire : le modele ne pourra pas etre flashe sur "
      "ce microcontroleur.")

# -----------------------------------------------------------------------------
# Point d'attention
#   Taille du fichier != taille en SRAM. La taille du .tflite correspond (a peu
#   pres) a la Flash requise. Le pic SRAM, lui, est invisible sans analyse du
#   graphe et des activations, comme le fait TinyEngine.
# -----------------------------------------------------------------------------

reflexion([
    "Notre analyseur simule un Memory Scheduler. Pourquoi le vrai TinyEngine "
    "obtient-il souvent un pic SRAM legerement inferieur a notre estimation ?",
    "Quel est l'avantage principal de generer du code C statique (TinyEngine) "
    "plutot que d'embarquer un interpreteur complet (TFLite Micro) sur le MCU ?",
])

next_step("exercises/05_what_if_it_doesnt_fit.py")
