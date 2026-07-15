#!/usr/bin/env python3
# =============================================================================
#  EXERCICE 03 : Pourquoi pas MobileNet ?  --  SOLUTION
# =============================================================================

import sys
sys.path.insert(0, "/workspace/scripts")
from coursekit import banner, section, save_fig, reflexion, next_step

banner("03", "Pourquoi pas MobileNet ? (SOLUTION)",
        "Profiler la memoire d'activation couche par couche", "45 min")

import copy
import torch
import matplotlib.pyplot as plt
from thop import profile
from mcunet.model_zoo import build_model

# -----------------------------------------------------------------------------
# 1. Chargement des modeles
# -----------------------------------------------------------------------------
section("1. Chargement des modeles")

mcunet_in2, _, _ = build_model(net_id="mcunet-in2", pretrained=False)
mbv2, _, _ = build_model(net_id="mbv2-w0.35", pretrained=False)
print("Modeles charges en memoire.")

# -----------------------------------------------------------------------------
# 2. Metriques statiques (MACs et parametres)
# -----------------------------------------------------------------------------
section("2. Metriques statiques (MACs et parametres)")

dummy_input_mcunet = torch.randn(1, 3, 160, 160)
dummy_input_mbv2 = torch.randn(1, 3, 144, 144)

macs_mcunet, params_mcunet = profile(mcunet_in2, inputs=(dummy_input_mcunet,), verbose=False)
macs_mbv2, params_mbv2 = profile(mbv2, inputs=(dummy_input_mbv2,), verbose=False)

print(f"MCUNet-in2  : {params_mcunet / 1e6:.2f} M Params | {macs_mcunet / 1e6:.1f} M MACs")
print(f"MobileNetV2 : {params_mbv2 / 1e6:.2f} M Params | {macs_mbv2 / 1e6:.1f} M MACs")

# -----------------------------------------------------------------------------
# 3. Profilage des activations
# -----------------------------------------------------------------------------
section("3. Profilage des activations")

activation_sizes = []

def hook_fn(module, inp, output):
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
# 4. Exercice : pic memoire et graphique  (SOLUTION)
# -----------------------------------------------------------------------------
section("4. Pic memoire et graphique")

peak_mbv2 = max(sizes_mbv2)
peak_mcunet = max(sizes_mcunet)

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

reflexion([
    "Pic memoire MobileNetV2 -> au tout debut : grande resolution spatiale "
    "(144x144, 72x72) meme avec peu de canaux, donc activations enormes. Ensuite "
    "la resolution baisse et les canaux augmentent, reduisant les activations.",
    "TinyNAS -> a bride l'expansion des premieres couches (bottlenecks etroits / "
    "downsampling plus rapide) pour ecreter le pic initial, en compensant par plus "
    "de profondeur/largeur dans les couches profondes (activations petites). D'ou "
    "plus de MACs au total : plus d'energie par inference, mais on rentre en SRAM.",
])

next_step("exercises/04_deployment_check.py")
