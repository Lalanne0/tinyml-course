#!/usr/bin/env python3
# =============================================================================
#  EXERCICE 02 : Inference en Python (INT8)  --  SOLUTION
# =============================================================================

import os
import sys
sys.path.insert(0, "/workspace/scripts")
from coursekit import banner, section, save_fig, reflexion, next_step

banner("02", "Inference en Python (INT8) (SOLUTION)",
        "Faire une inference TFLite int8 correcte (scale + zero_point)", "45 min")

import numpy as np
import tflite_runtime.interpreter as tflite
import matplotlib.pyplot as plt
from PIL import Image

# -----------------------------------------------------------------------------
# 1. Chargement du modele mcunet-vww1
# -----------------------------------------------------------------------------
section("1. Chargement du modele mcunet-vww1")

model_path = os.path.expanduser("~/.mcunet/mcunet-vww1.tflite")
print(f"Chargement du modele : {model_path}")
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
print("Modele alloue avec succes.")

# -----------------------------------------------------------------------------
# 2. Parametres de quantification
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
input_size = input_details["shape"][1:3]

# -----------------------------------------------------------------------------
# 3. Preprocessing d'une image
# -----------------------------------------------------------------------------
section("3. Preprocessing d'une image")

def get_sample_image(is_person=True):
    base_dir = "/dataset/vww_minival"
    target_dir = os.path.join(base_dir, "person" if is_person else "non_person")
    files = os.listdir(target_dir)
    return os.path.join(target_dir, files[0])

img_path = get_sample_image(is_person=True)
img = Image.open(img_path).convert("RGB")

img_resized = img.resize((input_size[1], input_size[0]))
input_data = np.expand_dims(img_resized, axis=0)
input_float = (np.float32(input_data) - 127.5) / 127.5
print(f"Plage float : [{np.min(input_float):.2f}, {np.max(input_float):.2f}]")

bad_input_int8 = input_float.astype(np.int8)
print("\n--- MAUVAISE QUANTIFICATION (simple cast) ---")
print(f"Valeurs         : {bad_input_int8[0, 0, 0]}")
print(f"Valeurs uniques : {len(np.unique(bad_input_int8))}")

# -----------------------------------------------------------------------------
# 4. Exercice : quantification correcte  (SOLUTION)
#    valeur_quantifiee = round(valeur_reelle / scale) + zero_point
# -----------------------------------------------------------------------------
section("4. Quantification correcte de l'entree")

good_input_int8 = np.round(input_float / input_scale) + input_zero_point
good_input_int8 = good_input_int8.astype(np.int8)

print("--- BONNE QUANTIFICATION ---")
print(f"Valeurs (echantillon) : {good_input_int8[0, 0, :5]}")
print(f"Valeurs uniques       : {len(np.unique(good_input_int8))}")

# -----------------------------------------------------------------------------
# 5. Inference
# -----------------------------------------------------------------------------
section("5. Inference")

interpreter.set_tensor(input_details["index"], good_input_int8)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details["index"])
print(f"Sortie brute (int8) : {output_data}")

# -----------------------------------------------------------------------------
# 6. Exercice : de-quantification  (SOLUTION)
#    valeur_reelle = scale * (valeur_quantifiee - zero_point)
# -----------------------------------------------------------------------------
section("6. De-quantification de la sortie")

output_scale, output_zero_point = output_details["quantization"]
probas_float = output_scale * (output_data.astype(np.float32) - output_zero_point)

print(f"Probabilites : {probas_float}")
classe_predite = np.argmax(probas_float[0])
classes = ["Non-personne", "Personne"]
print(f"Prediction finale : {classes[classe_predite]} "
      f"({probas_float[0][classe_predite] * 100:.1f}%)")

plt.imshow(img)
plt.title(f"Prediction : {classes[classe_predite]}")
plt.axis("off")
save_fig(plt, "02_prediction.png")

reflexion([
    "Normalisation [0,1] a l'entrainement mais [-1,1] a l'inference -> le modele "
    "percoit les entrees comme fortement decalees vers le negatif (un gris moyen "
    "0.5 devient 'noir' 0.0). Predictions degradees mais pas aleatoires.",
    "scale/zero_point differents entree/sortie -> les distributions different : "
    "entree = pixels dans [-1,1] (scale ~0.0078, zero_point ~0) ; sortie = logits "
    "d'une couche dense (plage plus large, distribution asymetrique). Ces "
    "parametres sont par tenseur, pas globaux.",
])

next_step("exercises/03_compare_vs_mobilenet.py")
