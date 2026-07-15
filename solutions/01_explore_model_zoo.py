#!/usr/bin/env python3
# =============================================================================
#  EXERCICE 01 : Exploration du Model Zoo  --  SOLUTION
# =============================================================================

import sys
sys.path.insert(0, "/workspace/scripts")
from coursekit import banner, section, reflexion, next_step

banner("01", "Exploration du Model Zoo (SOLUTION)",
        "Choisir le meilleur modele qui rentre dans un STM32F746", "30 min")

# -----------------------------------------------------------------------------
# 1. Budget memoire du STM32F746
# -----------------------------------------------------------------------------
section("1. Budget memoire du STM32F746")

TOTAL_SRAM_KB = 320
TOTAL_FLASH_KB = 1024
OS_RESERVE_SRAM_KB = 50
FW_RESERVE_FLASH_KB = 250

budget_sram = TOTAL_SRAM_KB - OS_RESERVE_SRAM_KB
budget_flash = TOTAL_FLASH_KB - FW_RESERVE_FLASH_KB

print(f"SRAM disponible pour le ML  : {budget_sram} KB")
print(f"Flash disponible pour le ML : {budget_flash} KB")

# -----------------------------------------------------------------------------
# 2. Modeles MCUNet-VWW disponibles
# -----------------------------------------------------------------------------
section("2. Modeles MCUNet-VWW disponibles")

import pandas as pd

vww_models_data = [
    {"net_id": "mcunet-vww0", "macs_m": 6.0,  "params_m": 0.37, "sram_kb": 146, "flash_kb": 617, "resolution": 64,  "acc_top1": 87.3},
    {"net_id": "mcunet-vww1", "macs_m": 11.6, "params_m": 0.43, "sram_kb": 162, "flash_kb": 689, "resolution": 80,  "acc_top1": 88.9},
    {"net_id": "mcunet-vww2", "macs_m": 55.8, "params_m": 0.64, "sram_kb": 311, "flash_kb": 897, "resolution": 144, "acc_top1": 91.8},
]

df_vww = pd.DataFrame(vww_models_data)
print(df_vww.to_string(index=False))

# -----------------------------------------------------------------------------
# 3. Exercice : selectionner le meilleur modele  (SOLUTION)
# -----------------------------------------------------------------------------
section("3. Exercice : selectionner le meilleur modele")

def select_best_model(models_df, max_sram, max_flash, min_acc):
    valid_models = models_df[
        (models_df["sram_kb"] <= max_sram)
        & (models_df["flash_kb"] <= max_flash)
        & (models_df["acc_top1"] >= min_acc)
    ]
    if valid_models.empty:
        return None
    sorted_models = valid_models.sort_values(by="acc_top1", ascending=False)
    return sorted_models.iloc[0]["net_id"]

best_model_id = select_best_model(df_vww, budget_sram, budget_flash, min_acc=85.0)
print(f"Le meilleur modele pour le STM32F746 est : {best_model_id}")
print("\nmcunet-vww2 n'est pas selectionne : il demande 311 KB de SRAM, "
      f"alors que le budget disponible n'est que de {budget_sram} KB.")

reflexion([
    "Pourquoi vww2 demande ~2x plus de SRAM que vww1 -> regarder la colonne "
    "'resolution' : les activations grandissent quadratiquement avec la "
    "resolution (64 -> 80 -> 144), les poids seulement lineairement.",
    "SRAM vs Flash -> vww2 sature la SRAM a 97% alors que la Flash a encore 12% "
    "de marge. C'est la SRAM qui determine la faisabilite en TinyML.",
])

next_step("exercises/02_inference_python.py")
