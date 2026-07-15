#!/usr/bin/env python3
# =============================================================================
#  Deploiement TinyML avec MCUNet
#  EXERCICE 01 : Exploration du Model Zoo
# =============================================================================
#
#  Objectif : Comprendre la structure du Model Zoo MCUNet et choisir un modele
#             adapte a un MCU cible selon ses contraintes de memoire.
#
#  Concepts cles :
#   1. SRAM  : memoire volatile rapide mais rare. Elle stocke les ACTIVATIONS
#              (features maps intermediaires) pendant l'inference.
#              C'est le goulot d'etranglement principal en TinyML.
#   2. Flash : memoire non volatile ou sont stockes le programme et les POIDS.
#   3. MACs  : Multiply-Accumulate Operations, mesure de la complexite de calcul.
#
#  Prerequis : avoir valide l'environnement (exercice 00).
# =============================================================================

import sys
sys.path.insert(0, "/workspace/scripts")
from coursekit import banner, section, todo_stop, reflexion, next_step

banner("01", "Exploration du Model Zoo",
        "Choisir le meilleur modele qui rentre dans un STM32F746", "30 min")

# -----------------------------------------------------------------------------
# 1. Definition de la cible Hardware : STM32F746 (ARM Cortex-M7, 216 MHz)
#    - SRAM totale  : 320 KB
#    - Flash totale : 1 MB (1024 KB)
#    Mais le modele ne peut pas tout utiliser : il faut de la place pour le
#    firmware, le RTOS et les buffers d'entree/sortie.
# -----------------------------------------------------------------------------
section("1. Budget memoire du STM32F746")

TOTAL_SRAM_KB = 320
TOTAL_FLASH_KB = 1024

OS_RESERVE_SRAM_KB = 50    # reserve pour l'OS / la pile
FW_RESERVE_FLASH_KB = 250  # reserve pour le firmware

budget_sram = TOTAL_SRAM_KB - OS_RESERVE_SRAM_KB
budget_flash = TOTAL_FLASH_KB - FW_RESERVE_FLASH_KB

print(f"SRAM disponible pour le ML  : {budget_sram} KB")
print(f"Flash disponible pour le ML : {budget_flash} KB")

# -----------------------------------------------------------------------------
# 2. Explorer les modeles MCUNet pour la tache VWW
#    Donnees issues du README officiel de MCUNet (quantification int8).
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
# 3. Petit exercice pratique
#    Ecrire une fonction select_best_model(...) qui retourne le net_id du
#    meilleur modele compatible : celui qui respecte les contraintes ET qui a
#    la plus grande precision (acc_top1).
# -----------------------------------------------------------------------------
section("3. Exercice : selectionner le meilleur modele")

def select_best_model(models_df, max_sram, max_flash, min_acc):
    """
    Selectionne le meilleur modele selon les contraintes.

    Args:
        models_df (pd.DataFrame): infos des modeles.
        max_sram (int)  : budget SRAM max en KB.
        max_flash (int) : budget Flash max en KB.
        min_acc (float) : precision Top-1 minimale requise.

    Returns:
        str | None : le net_id du modele choisi, ou None si aucun ne convient.
    """
    # ===== TODO ==============================================================
    #  1. Filtrer les modeles qui respectent max_sram, max_flash et min_acc.
    #  2. Trier les resultats par acc_top1 de maniere decroissante.
    #  3. Retourner le net_id du premier (le meilleur), ou None si liste vide.
    raise NotImplementedError("Completer la fonction select_best_model (exercice 01)")
    # =========================================================================

try:
    best_model_id = select_best_model(df_vww, budget_sram, budget_flash, min_acc=85.0)
except NotImplementedError:
    todo_stop("exercises/01_explore_model_zoo.py", "fonction select_best_model")

print(f"Le meilleur modele pour le STM32F746 est : {best_model_id}")
print("\nRemarque : mcunet-vww2 a une excellente precision. Avez-vous remarque "
      "pourquoi il n'est pas selectionne ? (Indice : comparer sa SRAM au budget.)")

# -----------------------------------------------------------------------------
# Point d'attention
#   La "taille" d'un modele n'est pas qu'une question de parametres. En TinyML,
#   ce sont souvent les ACTIVATIONS (SRAM) qui bloquent le deploiement en
#   premier, pas le nombre de poids.
# -----------------------------------------------------------------------------

reflexion([
    "Pourquoi mcunet-vww2 (311 KB SRAM) demande ~2x plus de SRAM que "
    "mcunet-vww1 (162 KB) alors que son nombre de parametres n'est superieur "
    "que de 50 % ? (Indice : regarder la colonne 'resolution'.)",
    "Entre SRAM et Flash, laquelle est generalement la plus restrictive (le "
    "goulot d'etranglement) en TinyML d'apres ce tableau ?",
])

next_step("exercises/02_inference_python.py")
