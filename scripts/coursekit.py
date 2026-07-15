"""
coursekit - petites fonctions d'affichage partagees par les exercices du cours.

Ce module n'a rien de "magique" : il sert juste a afficher de jolis bandeaux
dans le terminal, a arreter proprement un script tant qu'un TODO n'est pas
complete, et a sauvegarder les figures matplotlib (le container Docker n'a pas
d'ecran, on ne peut donc pas afficher une fenetre : on ecrit un fichier PNG).

Inutile de modifier ce fichier.
"""

import os
import sys

# Dossier partage avec votre machine hote (voir docker-compose.yml).
# Les figures generees par les exercices y sont ecrites.
OUTPUTS_DIR = "/workspace/outputs"

_L = 66  # largeur des bandeaux


def banner(num, titre, objectif, duree):
    """Affiche l'entete d'un exercice."""
    print("=" * _L)
    print(f"  EXERCICE {num} : {titre}")
    print("=" * _L)
    print(f"  Objectif : {objectif}")
    print(f"  Duree    : {duree}")
    print("=" * _L)


def section(titre):
    """Affiche un titre de section."""
    print(f"\n--- {titre} ---")


def todo_stop(fichier, repere):
    """
    Arrete le script proprement en expliquant OU aller completer le code.

    A appeler quand on detecte qu'un TODO n'a pas encore ete fait, pour eviter
    d'afficher une traceback Python inccomprehensible a l'apprenant.
    """
    print("\n" + "!" * _L)
    print("  /!\\  TODO A COMPLETER")
    print("!" * _L)
    print("  Ce script attend qu'une section marquee 'TODO' soit completee.")
    print()
    print(f"  1. Ouvrir le fichier :   nano {fichier}")
    print(f"  2. Trouver le bloc TODO ici :  {repere}")
    print("  3. Remplacer le placeholder par votre code.")
    print(f"  4. Relancer :            python {fichier}")
    print("!" * _L)
    sys.exit(0)


def save_fig(plt, nom):
    """Sauvegarde la figure matplotlib courante dans outputs/<nom>."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    chemin = os.path.join(OUTPUTS_DIR, nom)
    plt.savefig(chemin, bbox_inches="tight", dpi=110)
    plt.close("all")
    print(f"\n[figure] Sauvegardee : outputs/{nom}")
    print("         L'ouvrir depuis l'explorateur de fichiers de votre machine hote.")


def reflexion(questions):
    """Affiche les questions de reflexion de fin d'exercice."""
    print("\n" + "=" * _L)
    print("  QUESTIONS DE REFLEXION")
    print("=" * _L)
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q}")
    print("\n  (Les reponses se trouvent dans le dossier solutions/)")
    print("=" * _L)


def next_step(fichier_suivant):
    """Indique a l'apprenant l'exercice suivant (progression dans l'ordre)."""
    print("\n" + "=" * _L)
    if fichier_suivant is None:
        print("  Bravo, vous avez termine le dernier exercice du cours !")
    else:
        print("  Exercice termine.")
        print(f"  Etape suivante :   python {fichier_suivant}")
    print("=" * _L)
