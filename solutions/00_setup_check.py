#!/usr/bin/env python3
# =============================================================================
#  EXERCICE 00 : Verification de l'environnement  --  SOLUTION
# =============================================================================

import os
import sys

sys.path.insert(0, "/workspace/scripts")
from coursekit import banner, section, save_fig, reflexion, next_step

banner("00", "Verification de l'environnement (SOLUTION)",
        "Valider que Docker, les modeles et le dataset sont prets", "10 min")

# -----------------------------------------------------------------------------
# 1. Bibliotheques Python
# -----------------------------------------------------------------------------
section("1. Bibliotheques Python")

import tensorflow as tf
import torch
import torchvision
import numpy as np
import tflite_runtime
import mcunet

print(f"Python version      : {sys.version.split()[0]}")
print(f"TensorFlow version  : {tf.__version__}")
print(f"PyTorch version     : {torch.__version__}")
print(f"Torchvision version : {torchvision.__version__}")
print(f"NumPy version       : {np.__version__}")
print(f"TFLite Runtime      : {tflite_runtime.__version__}")
print("MCUNet importe avec succes !")

# -----------------------------------------------------------------------------
# 2. Modeles pre-telecharges
# -----------------------------------------------------------------------------
section("2. Modeles pre-telecharges")

from mcunet.model_zoo import net_id_list

print("Modeles disponibles dans le MCUNet Model Zoo :")
for net_id in net_id_list:
    print(f" - {net_id}")

print("\nVerification du cache local des modeles TFLite...")
cache_dir = os.path.expanduser("~/.mcunet/")
if os.path.exists(cache_dir):
    tflite_files = [f for f in os.listdir(cache_dir) if f.endswith(".tflite")]
    print(f"Trouve {len(tflite_files)} modeles TFLite dans le cache :")
    for f in tflite_files:
        size_kb = os.path.getsize(os.path.join(cache_dir, f)) / 1024
        print(f" - {f} ({size_kb:.1f} KB)")
else:
    print("Dossier de cache non trouve. Les modeles seront telecharges a la volee.")

# -----------------------------------------------------------------------------
# 3. Dataset VWW
# -----------------------------------------------------------------------------
section("3. Dataset VWW (Visual Wake Words)")

data_dir = "/dataset/vww_minival"
person_dir = os.path.join(data_dir, "person")
non_person_dir = os.path.join(data_dir, "non_person")

if os.path.exists(data_dir):
    n_person = len(os.listdir(person_dir)) if os.path.exists(person_dir) else 0
    n_non_person = len(os.listdir(non_person_dir)) if os.path.exists(non_person_dir) else 0
    print("Dataset VWW minival trouve !")
    print(f"Images 'Person'     : {n_person}")
    print(f"Images 'Non-Person' : {n_non_person}")
    print(f"Total               : {n_person + n_non_person} images")
else:
    print("ERREUR : Dataset non trouve. Verifier le build Docker.")

# -----------------------------------------------------------------------------
# 4. Exercice : charger une image au hasard  (SOLUTION)
# -----------------------------------------------------------------------------
section("4. Exercice : charger une image au hasard")

import matplotlib.pyplot as plt
from PIL import Image
import random

files = os.listdir(person_dir)
random_file = random.choice(files)
img = Image.open(os.path.join(person_dir, random_file))

plt.imshow(img)
plt.title(f"Exemple Visual Wake Words (Personne)\n{random_file}")
plt.axis("off")
save_fig(plt, "00_sample_person.png")

# -----------------------------------------------------------------------------
# Questions de reflexion  --  reponses
# -----------------------------------------------------------------------------
reflexion([
    "Taille du modele vs dataset -> Le modele a ete entraine sur beaucoup de "
    "donnees, mais on n'en garde qu'une liste de parametres, bien plus legere. "
    "On peut en plus quantifier les poids, reduisant encore la taille sur disque.",
    "Compatibilite ascendante -> MCUNet a ete developpe en 2019-2020 (ere TF 1.x). "
    "L'API et le comportement de la conversion TFLite int8 different entre TF1 et "
    "TF2 (graphes statiques vs eager, operateurs quantifies supportes, format "
    "binaire des modeles). D'ou l'importance d'utiliser exactement TF 1.15.",
])

next_step("exercises/01_explore_model_zoo.py")
