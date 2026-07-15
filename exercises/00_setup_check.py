#!/usr/bin/env python3
# =============================================================================
#  Deploiement TinyML avec MCUNet
#  EXERCICE 00 : Verification de l'environnement
# =============================================================================
#
#  Objectif : Verifier que l'environnement Docker fonctionne et que toutes les
#             ressources (donnees, modeles, bibliotheques) sont disponibles.
#
#  Concepts cles :
#   1. TinyML   : executer du Machine Learning sur des ressources tres
#                 contraintes (microcontroleurs).
#   2. MCUNet   : framework co-concu (systeme + algorithme) par le MIT.
#   3. Env fige : le TinyML repose sur des versions tres specifiques de
#                 bibliotheques (ici TensorFlow 1.15) pour la compatibilite
#                 avec les outils d'export vers C/C++.
#
#  Comment travailler sur ce fichier :
#   - Lire les commentaires de haut en bas (c'est votre "cours").
#   - Lancer le script :   python exercises/00_setup_check.py
#   - Completer les blocs marques "TODO", puis relancer.
# =============================================================================

import os
import sys

# Rend le module partage "coursekit" importable (il vit dans scripts/).
sys.path.insert(0, "/workspace/scripts")
from coursekit import banner, section, todo_stop, save_fig, reflexion, next_step

banner("00", "Verification de l'environnement",
        "Valider que Docker, les modeles et le dataset sont prets", "10 min")

# -----------------------------------------------------------------------------
# 1. Verification des bibliotheques Python
#    On s'assure que les versions demandees par MCUNet sont bien installees.
#    (TensorFlow 1.15 affiche des warnings de depreciation : c'est normal.)
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
# 2. Verification des modeles pre-telecharges
#    Le Dockerfile a normalement mis en cache plusieurs modeles TFLite.
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
# 3. Verification du Dataset VWW (Visual Wake Words)
#    VWW est le "Hello World" de la vision sur microcontroleur : detecter si une
#    personne est presente ou non sur une petite image.
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
# 4. Petit exercice pratique
#    Charger une image au hasard de la classe "person" et la sauvegarder.
# -----------------------------------------------------------------------------
section("4. Exercice : charger une image au hasard")

import matplotlib.pyplot as plt
from PIL import Image
import random

# ===== TODO ==================================================================
#  1. Choisir un fichier au hasard dans person_dir (random.choice + os.listdir).
#  2. L'ouvrir avec Image.open(...).
#  3. Assigner l'image ouverte a la variable `img`.
img = None  # <-- remplacer None par votre code
# =============================================================================

if img is None:
    todo_stop("exercises/00_setup_check.py", "section 4 (variable `img`)")

plt.imshow(img)
plt.title("Exemple Visual Wake Words (Personne)")
plt.axis("off")
save_fig(plt, "00_sample_person.png")

# -----------------------------------------------------------------------------
# Questions de reflexion
# -----------------------------------------------------------------------------
reflexion([
    "Taille du modele vs dataset : les modeles TinyML font < 500 KB, pourtant "
    "le dataset d'evaluation (minival) fait 380 MB. Pourquoi le dataset est-il "
    "si volumineux par rapport au modele final ?",
    "Compatibilite ascendante : pourquoi est-il critique d'utiliser TensorFlow "
    "1.15 et non une version recente (ex. TF 2.15) pour l'export vers MCU ?",
])

next_step("exercises/01_explore_model_zoo.py")
