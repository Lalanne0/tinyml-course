#!/usr/bin/env python3
# =============================================================================
#  Deploiement TinyML avec MCUNet
#  EXERCICE 02 : Inference en Python (INT8)
# =============================================================================
#
#  Objectif : Charger un modele TFLite quantifie, comprendre les parametres de
#             quantification, et realiser une inference correcte en gerant
#             l'echelle (scale) et le point zero (zero_point).
#
#  Concepts cles :
#   1. Quantification INT8 : convertit poids et activations float32 en entiers
#      8 bits. Divise la taille du modele par 4 et accelere l'inference sur MCU.
#   2. Equation de quantification :
#         valeur_reelle = scale * (valeur_quantifiee - zero_point)
#   3. TFLite Interpreter : moteur d'execution qui alloue les tenseurs et
#      invoque l'inference.
#
#  Prerequis : notions de types de donnees (float32 vs int8), exercice 01 fait.
# =============================================================================

import os
import sys
sys.path.insert(0, "/workspace/scripts")
from coursekit import banner, section, todo_stop, save_fig, reflexion, next_step

banner("02", "Inference en Python (INT8)",
        "Faire une inference TFLite int8 correcte (scale + zero_point)", "45 min")

import numpy as np
import tflite_runtime.interpreter as tflite
import matplotlib.pyplot as plt
from PIL import Image

# -----------------------------------------------------------------------------
# 1. Chargement du modele selectionne
#    A l'exercice 01, mcunet-vww1 etait le meilleur choix pour le STM32F746.
# -----------------------------------------------------------------------------
section("1. Chargement du modele mcunet-vww1")

model_path = os.path.expanduser("~/.mcunet/mcunet-vww1.tflite")
print(f"Chargement du modele : {model_path}")

interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
print("Modele alloue avec succes.")

# -----------------------------------------------------------------------------
# 2. Inspection des tenseurs d'entree / sortie
#    Pour un modele quantifie, il est INDISPENSABLE de recuperer ses parametres
#    de quantification, sinon impossible de preparer l'image correctement.
# -----------------------------------------------------------------------------
section("2. Parametres de quantification")

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print("--- Entree ---")
print(f"Shape          : {input_details['shape']}")
print(f"Type           : {input_details['dtype']}")
print(f"Quantification : {input_details['quantization']}")

print("--- Sortie ---")
print(f"Shape          : {output_details['shape']}")
print(f"Type           : {output_details['dtype']}")
print(f"Quantification : {output_details['quantization']}")

input_scale, input_zero_point = input_details["quantization"]
input_size = input_details["shape"][1:3]  # (hauteur, largeur)

# -----------------------------------------------------------------------------
# 3. Preprocessing d'une image
#    Pour etre acceptee par le modele, une image doit etre :
#     1. redimensionnee a la bonne taille ;
#     2. normalisee dans la plage d'origine (ici [-1, 1] pour simplifier) ;
#     3. quantifiee en int8.
# -----------------------------------------------------------------------------
section("3. Preprocessing d'une image")

def get_sample_image(is_person=True):
    base_dir = "/dataset/vww_minival"
    target_dir = os.path.join(base_dir, "person" if is_person else "non_person")
    files = os.listdir(target_dir)
    return os.path.join(target_dir, files[0])  # on prend la premiere

img_path = get_sample_image(is_person=True)
img = Image.open(img_path).convert("RGB")

# 1. Redimensionnement
img_resized = img.resize((input_size[1], input_size[0]))
input_data = np.expand_dims(img_resized, axis=0)

# 2. Normalisation float [-1, 1]
input_float = (np.float32(input_data) - 127.5) / 127.5
print(f"Plage float : [{np.min(input_float):.2f}, {np.max(input_float):.2f}]")

# --- L'exercice piege : la MAUVAISE quantification (un simple cast) ---
bad_input_int8 = input_float.astype(np.int8)
print("\n--- MAUVAISE QUANTIFICATION (simple cast) ---")
print(f"Valeurs             : {bad_input_int8[0, 0, 0]}")
print(f"Valeurs uniques     : {len(np.unique(bad_input_int8))}")
print("Toutes les valeurs sont ecrasees en 0 ou -1 : l'information est detruite !")

# -----------------------------------------------------------------------------
# 4. Exercice : la BONNE quantification
#    Rappel :  valeur_quantifiee = round(valeur_reelle / scale) + zero_point
# -----------------------------------------------------------------------------
section("4. Exercice : quantification correcte de l'entree")

# ===== TODO ==================================================================
#  Calculer l'entree quantifiee correctement avec input_scale et input_zero_point.
#  Ne pas oublier d'arrondir (np.round) et de caster en np.int8 a la fin.
good_input_int8 = np.zeros_like(input_float, dtype=np.int8)  # <-- remplacer cette ligne
# =============================================================================

if np.all(good_input_int8 == 0):
    todo_stop("exercises/02_inference_python.py", "section 4 (good_input_int8)")

print("--- BONNE QUANTIFICATION ---")
print(f"Valeurs (echantillon) : {good_input_int8[0, 0, :5]}")
print(f"Valeurs uniques       : {len(np.unique(good_input_int8))}")

# -----------------------------------------------------------------------------
# 5. Execution de l'inference
# -----------------------------------------------------------------------------
section("5. Inference")

interpreter.set_tensor(input_details["index"], good_input_int8)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details["index"])
print(f"Sortie brute (int8) : {output_data}")

# -----------------------------------------------------------------------------
# 6. Exercice : de-quantifier la sortie
#    La sortie est un entier quantifie. On la reconvertit en probabilites
#    float32 avec la meme formule, a l'envers :
#         valeur_reelle = scale * (valeur_quantifiee - zero_point)
# -----------------------------------------------------------------------------
section("6. Exercice : de-quantification de la sortie")

output_scale, output_zero_point = output_details["quantization"]

# ===== TODO ==================================================================
#  De-quantifier output_data pour obtenir un tableau de probabilites en float32.
probas_float = np.array([[0.0, 0.0]])  # <-- remplacer cette ligne
# =============================================================================

if np.all(probas_float == 0):
    todo_stop("exercises/02_inference_python.py", "section 6 (probas_float)")

print(f"Probabilites : {probas_float}")
classe_predite = np.argmax(probas_float[0])
classes = ["Non-personne", "Personne"]
print(f"Prediction finale : {classes[classe_predite]} "
      f"({probas_float[0][classe_predite] * 100:.1f}%)")

plt.imshow(img)
plt.title(f"Prediction : {classes[classe_predite]}")
plt.axis("off")
save_fig(plt, "02_prediction.png")

# -----------------------------------------------------------------------------
# Point d'attention (l'erreur n1 en deploiement TFLite int8)
#   Ne faire JAMAIS un simple img.astype(np.int8). Oublier scale/zero_point
#   donne des resultats qui ont l'air aleatoires : vous croirez que le modele est
#   casse, alors que c'est seulement votre pretraitement qui est mauvais.
# -----------------------------------------------------------------------------

reflexion([
    "Que se passerait-il si le modele etait entraine avec une normalisation "
    "[0, 1] mais que vous fournissiez des images normalisees [-1, 1] avant la "
    "quantification ? Comment le modele reagirait-il ?",
    "Pourquoi les valeurs de scale et zero_point sont-elles differentes pour "
    "l'entree et pour la sortie ?",
])

next_step("exercises/03_compare_vs_mobilenet.py")
