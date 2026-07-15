#!/usr/bin/env python3
# =============================================================================
#  EXERCICE 05 : Et si ca ne rentre pas ?  --  SOLUTION
# =============================================================================

import os
import sys
sys.path.insert(0, "/workspace/scripts")
from coursekit import banner, section, reflexion, next_step
from tinyengine_analysis import analyze_tflite, print_report

banner("05", "Et si ca ne rentre pas ? (SOLUTION)",
        "Arbitrer entre les strategies quand le modele est trop gros", "30 min")

# -----------------------------------------------------------------------------
# Strategie 1 : Downgrade du modele
# -----------------------------------------------------------------------------
section("Strategie 1 : Downgrade du modele")
print("Choisir mcunet-vww1 (162 KB SRAM, 88.9%) au lieu de vww2.")
print("Impact : -2.9% de precision, 0 cout de dev, 0 cout materiel.")

# -----------------------------------------------------------------------------
# Strategie 2 : Upgrade Hardware (STM32H743)
# -----------------------------------------------------------------------------
section("Strategie 2 : Upgrade Hardware (STM32H743)")

tflite_vww2 = os.path.expanduser("~/.mcunet/mcunet-vww2.tflite")
result_vww2 = analyze_tflite(tflite_vww2)
print_report(result_vww2, "STM32H743")
print("\nImpact : precision maintenue (91.8%), mais +2 M$ sur 1 M d'unites.")

# -----------------------------------------------------------------------------
# Strategie 3 : Reduction de la resolution (144 -> 96)  (SOLUTION)
# -----------------------------------------------------------------------------
section("Strategie 3 : Reduction de la resolution (144 -> 96)")

original_resolution = 144
new_resolution = 96
original_sram_kb = result_vww2["sram_peak_kb"]

ratio = (new_resolution ** 2) / (original_resolution ** 2)
new_sram_kb = original_sram_kb * ratio

print(f"Resolution originale : {original_resolution}x{original_resolution}")
print(f"Nouvelle resolution  : {new_resolution}x{new_resolution}")
print(f"Ratio des aires      : {ratio:.2f}")
print(f"Pic SRAM original    : {original_sram_kb:.1f} KB")
print(f"Nouveau pic estime   : {new_sram_kb:.1f} KB")

if new_sram_kb < 270:
    print("\n=> Ca rentre sur le STM32F746 !")
else:
    print("\n=> Toujours trop grand.")

print("\nImpact : necessite de reentrainer / fine-tuner. Baisse probable de "
      "precision. Temps de dev : moyen/haut.")

# -----------------------------------------------------------------------------
# Strategie 4 : Patch-Based Inference (MCUNetV2)
# -----------------------------------------------------------------------------
section("Strategie 4 : Patch-Based Inference (MCUNetV2)")
print("Decoupe l'image en patchs, execute les premieres couches patch par patch,")
print("puis recombine. Pic SRAM initial / 4, au prix de ~10% de MACs en plus.")

reflexion([
    "Strategie 1 vs 2 -> c'est un calcul de ROI : chiffrer le surcout materiel "
    "(delta cout unitaire x volume) contre le cout d'un modele moins precis "
    "(faux positifs, SAV, experience utilisateur). Sur de gros volumes on reste "
    "souvent sur le materiel le moins cher.",
    "Patchs (calcul contre memoire) -> le MCU n'a PAS plus de SRAM : si ca "
    "depasse, le programme crashe (HardFault) ou ne compile pas, sans swap "
    "possible. Le calcul, lui, ne coute que du temps : +10% d'inference (ex. "
    "200 -> 220 ms) est souvent acceptable, la ou manquer de memoire est fatal.",
])

next_step(None)
