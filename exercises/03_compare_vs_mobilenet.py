#!/usr/bin/env python3
# =============================================================================
#  Deploiement TinyML avec MCUNet
#  EXERCICE 03 : Pourquoi pas MobileNet ?
# =============================================================================
#
#  Objectif : Comprendre concretement pourquoi un modele "mobile" classique
#             comme MobileNetV2 ne convient pas a un MCU, en analysant la
#             repartition de la memoire couche par couche.
#
#  Concepts cles :
#   1. MobileNetV2  : reference pour les telephones (Edge AI). Efficace en calcul
#      (MACs) mais pas optimise pour les pics de SRAM.
#   2. Peak Memory  : pendant l'inference couche par couche, la SRAM stocke
#      l'entree ET la sortie de la couche. Le max de cette taille = "Pic SRAM".
#   3. THOP         : outil PyTorch pour compter MACs et parametres.
#
#  Prerequis : notions d'architecture CNN (convolutions, inverted residuals).
# =============================================================================

import sys
sys.path.insert(0, "/workspace/scripts")
from coursekit import banner, section, todo_stop, save_fig, reflexion, next_step

banner("03", "Pourquoi pas MobileNet ?",
        "Profiler la memoire d'activation couche par couche", "45 min")

import copy
import torch
import matplotlib.pyplot as plt
from thop import profile
from mcunet.model_zoo import build_model

# -----------------------------------------------------------------------------
# 1. Chargement des modeles a comparer
#    Deux modeles ImageNet de taille (poids) similaire : MobileNetV2-w0.35 et
#    MCUNet-in2.
# -----------------------------------------------------------------------------
section("1. Chargement des modeles")

mcunet_in2, _, _ = build_model(net_id="mcunet-in2", pretrained=False)
mbv2, _, _ = build_model(net_id="mbv2-w0.35", pretrained=False)
print("Modeles charges en memoire.")

# -----------------------------------------------------------------------------
# 2. Comparaison des metriques statiques (poids et calcul) avec THOP
# -----------------------------------------------------------------------------
section("2. Metriques statiques (MACs et parametres)")

dummy_input_mcunet = torch.randn(1, 3, 160, 160)  # mcunet-in2 : 160x160
dummy_input_mbv2 = torch.randn(1, 3, 144, 144)    # mbv2-w0.35 : 144x144

macs_mcunet, params_mcunet = profile(mcunet_in2, inputs=(dummy_input_mcunet,), verbose=False)
macs_mbv2, params_mbv2 = profile(mbv2, inputs=(dummy_input_mbv2,), verbose=False)

print(f"MCUNet-in2  : {params_mcunet / 1e6:.2f} M Params | {macs_mcunet / 1e6:.1f} M MACs")
print(f"MobileNetV2 : {params_mbv2 / 1e6:.2f} M Params | {macs_mbv2 / 1e6:.1f} M MACs")
print("\nEtonnant : MCUNet a ~3x plus de MACs pour un nombre de parametres "
      "similaire. Le vrai probleme n'est pas le calcul (ca prend du temps),\n"
      "c'est la SRAM (ca crashe si on depasse).")

# -----------------------------------------------------------------------------
# 3. Profilage de la memoire d'activation (SRAM)
#    On utilise des forward hooks PyTorch pour enregistrer la taille des
#    tenseurs de sortie de chaque couche Conv2d pendant une inference.
# -----------------------------------------------------------------------------
section("3. Profilage des activations")

activation_sizes = []

def hook_fn(module, inp, output):
    # Taille du tenseur de sortie en KB (en supposant int8 = 1 byte).
    activation_sizes.append(output.numel() * 1 / 1024)

def profile_activations(model, dummy_input):
    global activation_sizes
    activation_sizes = []
    handles = []
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            handles.append(module.register_forward_hook(hook_fn))
    with torch.no_grad():
        model(dummy_input)
    for handle in handles:
        handle.remove()
    return copy.deepcopy(activation_sizes)

sizes_mcunet = profile_activations(mcunet_in2, dummy_input_mcunet)
sizes_mbv2 = profile_activations(mbv2, dummy_input_mbv2)
print(f"Profilage termine : {len(sizes_mcunet)} couches Conv pour MCUNet, "
      f"{len(sizes_mbv2)} pour MobileNetV2.")

# -----------------------------------------------------------------------------
# 4. Exercice pratique
#    Trouver le pic memoire de chaque modele et tracer la comparaison.
# -----------------------------------------------------------------------------
section("4. Exercice : pic memoire et graphique")

# ===== TODO ==================================================================
#  Trouver le pic memoire (max) de sizes_mbv2 et de sizes_mcunet.
peak_mbv2 = 0     # <-- remplacer par le max de sizes_mbv2
peak_mcunet = 0   # <-- remplacer par le max de sizes_mcunet
# =============================================================================

if peak_mbv2 == 0 or peak_mcunet == 0:
    todo_stop("exercises/03_compare_vs_mobilenet.py", "section 4 (peak_mbv2, peak_mcunet)")

print(f"Pic SRAM MobileNetV2 : {peak_mbv2:.1f} KB")
print(f"Pic SRAM MCUNet-in2  : {peak_mcunet:.1f} KB")

plt.figure(figsize=(10, 5))
plt.plot(sizes_mbv2, label=f"MobileNetV2-w0.35 (Pic: {peak_mbv2:.1f} KB)", marker="o", markersize=3)
plt.plot(sizes_mcunet, label=f"MCUNet-in2 (Pic: {peak_mcunet:.1f} KB)", marker="x", markersize=3)
plt.title("Taille des activations par couche (empreinte SRAM)")
plt.xlabel("Index de la couche de convolution")
plt.ylabel("Taille de l'activation (KB)")
plt.grid(True, alpha=0.3)
plt.legend()
save_fig(plt, "03_activation_profile.png")

# -----------------------------------------------------------------------------
# Point d'attention
#   Ce calcul est simplifie : le vrai moteur (TinyEngine) doit stocker l'entree
#   ET la sortie simultanement (ping-pong buffers) plus des variables de travail.
#   Le vrai pic SRAM est donc plus eleve, mais la forme de la courbe reste la meme.
# -----------------------------------------------------------------------------

reflexion([
    "Dans quelle partie du reseau se trouve le pic memoire de MobileNetV2 "
    "(debut, milieu, fin) ? Pourquoi (taille spatiale vs nombre de canaux) ?",
    "MCUNet a plus de MACs mais un pic memoire plus bas. Comment TinyNAS a-t-il "
    "modifie l'architecture pour cela, et qu'est-ce que ca implique pour la "
    "consommation energetique ?",
])

next_step("exercises/04_deployment_check.py")
