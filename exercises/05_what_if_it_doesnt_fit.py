#!/usr/bin/env python3
# =============================================================================
#  Deploiement TinyML avec MCUNet
#  EXERCICE 05 : Et si ca ne rentre pas ?
# =============================================================================
#
#  Objectif : Comprendre et arbitrer entre les strategies d'optimisation quand
#             un modele depasse le budget memoire du MCU.
#
#  Le probleme : on veut la haute precision de mcunet-vww2 (91.8%), mais il
#  demande > 310 KB de SRAM, alors que le STM32F746 n'a que ~270 KB disponibles
#  pour le ML. Quelles sont nos options en tant qu'ingenieur Edge AI ?
#
#  Prerequis : avoir fait l'analyse de deploiement (exercice 04).
# =============================================================================

import os
import sys
sys.path.insert(0, "/workspace/scripts")
from coursekit import banner, section, todo_stop, reflexion, next_step
from tinyengine_analysis import analyze_tflite, print_report

banner("05", "Et si ca ne rentre pas ?",
        "Arbitrer entre les strategies quand le modele est trop gros", "30 min")

# -----------------------------------------------------------------------------
# Strategie 1 : Downgrade du modele (le compromis logiciel)
#   Choisir un modele plus petit du Model Zoo. mcunet-vww1 rentre (162 KB).
#   Impact : precision 91.8% -> 88.9% (-2.9%). Temps de dev : 0. Cout HW : 0.
# -----------------------------------------------------------------------------
section("Strategie 1 : Downgrade du modele")
print("Choisir mcunet-vww1 (162 KB SRAM, 88.9%) au lieu de vww2.")
print("Impact : -2.9% de precision, 0 cout de dev, 0 cout materiel.")

# -----------------------------------------------------------------------------
# Strategie 2 : Upgrade Hardware (la force brute)
#   Passer au STM32H743 (512 KB SRAM, 2 MB Flash). On garde vww2 et sa precision.
# -----------------------------------------------------------------------------
section("Strategie 2 : Upgrade Hardware (STM32H743)")

tflite_vww2 = os.path.expanduser("~/.mcunet/mcunet-vww2.tflite")
result_vww2 = analyze_tflite(tflite_vww2)
print_report(result_vww2, "STM32H743")
print("\nImpact : precision maintenue (91.8%), mais cout unitaire superieur.")
print("Sur 1 M d'unites, passer de 6$ a 8$ = +2 M$ pour l'entreprise.")

# -----------------------------------------------------------------------------
# Strategie 3 : Reduction de la resolution d'entree (le hack ingenieux)
#   mcunet-vww2 consomme tant de SRAM surtout a cause de sa resolution 144x144
#   (contre 80x80 pour vww1). La taille des activations au debut du reseau
#   evolue quadratiquement avec la resolution.
# -----------------------------------------------------------------------------
section("Strategie 3 : Reduction de la resolution (144 -> 96)")

original_resolution = 144
new_resolution = 96
original_sram_kb = result_vww2["sram_peak_kb"]

# ===== TODO ==================================================================
#  On suppose le pic memoire proportionnel au nombre de pixels de l'entree.
#  1. Calculer le ratio de reduction de l'aire (pixels) : (new^2) / (orig^2).
#  2. Calculer le nouveau pic SRAM estime : original_sram_kb * ratio.
ratio = 1.0        # <-- remplacer par le vrai ratio des aires
new_sram_kb = 0.0  # <-- remplacer par la nouvelle estimation SRAM
# =============================================================================

if new_sram_kb == 0.0:
    todo_stop("exercises/05_what_if_it_doesnt_fit.py", "section Strategie 3 (ratio, new_sram_kb)")

print(f"Resolution originale : {original_resolution}x{original_resolution}")
print(f"Nouvelle resolution  : {new_resolution}x{new_resolution}")
print(f"Ratio des aires      : {ratio:.2f}")
print(f"Pic SRAM original    : {original_sram_kb:.1f} KB")
print(f"Nouveau pic estime   : {new_sram_kb:.1f} KB")

if new_sram_kb < 270:
    print("\n=> Ca rentre sur le STM32F746 !")
else:
    print("\n=> Toujours trop grand.")

print("\nImpact : necessite de reentrainer / fine-tuner le modele a la nouvelle "
      "resolution. Baisse probable de precision. Temps de dev : moyen/haut.")

# -----------------------------------------------------------------------------
# Strategie 4 : Patch-Based Inference (MCUNetV2)
#   La solution avancee du MIT : au lieu d'inferer toute l'image d'un coup (gros
#   tenseur sur les premieres couches), TinyEngine decoupe l'image en "patchs"
#   et execute les premieres couches un patch a la fois. Les features maps
#   (plus petites) sont recombinees avant les couches fully connected.
#   -> divise le pic SRAM initial par ~4 sans perte d'information, +~10% de MACs.
# -----------------------------------------------------------------------------
section("Strategie 4 : Patch-Based Inference (MCUNetV2)")
print("Decoupe l'image en patchs, execute les premieres couches patch par patch,")
print("puis recombine. Pic SRAM initial / 4, au prix de ~10% de MACs en plus.")

reflexion([
    "Votre modele est a 30 KB au-dessus du budget SRAM. Comment choisir entre "
    "la strategie 1 (modele plus petit) et la strategie 2 (hardware plus "
    "puissant) ? Quels facteurs prendre en compte ?",
    "La technique des patchs (MCUNetV2) echange du calcul (plus de MACs) contre "
    "de la memoire (moins de SRAM). Pourquoi cet echange est-il generalement "
    "gagnant sur un microcontroleur ?",
])

next_step(None)  # dernier exercice du cours
