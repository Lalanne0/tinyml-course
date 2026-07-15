# TinyML avec MCUNet

Cas d'etude pratique pour deployer des modeles de deep learning sur microcontroleurs avec le framework MCUNet du MIT.

**Tache fil rouge :** Visual Wake Words (VWW) - classification binaire "personne presente / absente" sur des images 80x80, ciblee sur un STM32F746.

Le cours se fait entierement en **scripts Python** que l'on execute et complete dans un terminal (pas de Jupyter).

## Prerequis

- **Docker Desktop** (Mac/Windows) ou **Docker Engine** (Linux)
- **8 GB de RAM** libre minimum
- **10 GB de disque** libre (image Docker + dataset VWW)

> **Apple Silicon (M1/M2/M3) :** TensorFlow 1.15 n'a pas de wheel ARM64 native. Ajouter `--platform=linux/amd64` dans le `docker-compose.yml`. L'execution sera plus lente via emulation.

## Demarrage rapide

```bash
git clone <url-du-repo>
cd tinyml-course
make start      # construit l'image et lance le container en arriere-plan (peut prendre jusqu'à 10 minutes)
make shell      # ouvre un terminal dans le container (instantané, affiche le guide)
```

Une fois dans le shell du container, lancer le premier exercice :

```bash
python exercises/00_setup_check.py
```

## Comment travailler

Chaque exercice est un script dans `exercises/`. La boucle de travail est toujours la meme :

1. **Lancer** le script : `python exercises/00_setup_check.py`
2. S'il affiche un bandeau **`TODO A COMPLETER`**, ouvrir le fichier avec un editeur :
   ```bash
   nano exercises/00_setup_check.py
   ```
   (`Ctrl+O` = sauver, `Ctrl+X` = quitter ; `vim` est aussi disponible)
3. **Completer** le bloc marque `# ===== TODO =====`, sauvegarder, puis **relancer**.
4. Quand tout est bon, le script vous indique **l'exercice suivant**.

Les exercices sont concus pour etre faits **dans l'ordre** (00 -> 05).

> Les fichiers de `exercises/` et `outputs/` sont montes depuis votre machine : vos modifications et les figures generees sont conservees sur l'hote, meme apres `make stop`.

### Figures

Le container n'a pas d'ecran : les scripts qui produisent un graphique l'ecrivent dans `outputs/` (au format PNG). Ouvrir ces fichiers depuis l'explorateur de fichiers de votre machine hote (dossier `outputs/` du projet).

### Bloque ?

Les corrections completes sont dans `solutions/` (lecture seule). Par exemple :

```bash
python solutions/00_setup_check.py
```

## Plan des exercices

| # | Fichier | Objectif | Duree estimee |
|---|---------|----------|---------------|
| 00 | `exercises/00_setup_check.py` | Verifier l'environnement | 10 min |
| 01 | `exercises/01_explore_model_zoo.py` | Explorer le Model Zoo MCUNet | 30 min |
| 02 | `exercises/02_inference_python.py` | Inference TFLite int8 avec quantification | 45 min |
| 03 | `exercises/03_compare_vs_mobilenet.py` | Comparer MCUNet vs MobileNetV2 | 45 min |
| 04 | `exercises/04_deployment_check.py` | Analyse de deploiement avec TinyEngine | 40 min |
| 05 | `exercises/05_what_if_it_doesnt_fit.py` | Strategies quand le modele ne tient pas | 30 min |

**Duree totale estimee :** environ 3h30.

## Architecture du repo

```
tinyml-course/
|-- docker/
|   |-- Dockerfile             # Image Python 3.7 + TF 1.15 + MCUNet (+ nano/vim)
|   +-- docker-compose.yml     # Configuration du service
|-- exercises/                  # Scripts a completer par l'apprenant
|   |-- 00_setup_check.py
|   |-- 01_explore_model_zoo.py
|   |-- 02_inference_python.py
|   |-- 03_compare_vs_mobilenet.py
|   |-- 04_deployment_check.py
|   +-- 05_what_if_it_doesnt_fit.py
|-- solutions/                  # Scripts avec solutions completes (lecture seule)
|-- scripts/
|   |-- coursekit.py            # Fonctions d'affichage partagees (bandeaux, TODO, figures)
|   +-- tinyengine_analysis.py  # Wrapper Python pour TinyEngine
|-- outputs/                    # Figures generees par les exercices (non versionne)
|-- data/                       # Peuple par Docker (non versionne)
|-- START_HERE.txt              # Guide affiche a l'ouverture du shell
|-- Makefile                    # Commandes : start, shell, run, check, stop, clean, logs
+-- README.md
```

## Commandes disponibles

| Commande | Description |
|----------|-------------|
| `make start` | Construit l'image et lance le container en arriere-plan |
| `make shell` | Ouvre un terminal dans le container (affiche le guide) |
| `make run EX=02` | Execute un exercice sans entrer dans le shell |
| `make check` | Execute l'exercice 00 pour verifier l'environnement |
| `make stop` | Arrete le container |
| `make clean` | Supprime container, image et volumes |
| `make logs` | Affiche les logs du container |

## Troubleshooting

### Docker manque de memoire

Augmenter la RAM allouee a 8 GB minimum dans Docker Desktop > Settings > Resources.

### `make shell` renvoie une erreur "container not running"

Lancer d'abord `make start`, puis `make shell`.

### Build trop long

C'est normal au premier lancement (~15-20 min). Le dataset VWW fait 380 MB et les modeles sont pre-telecharges. Les builds suivants seront quasi instantanes grace au cache Docker.

### Erreur d'import mcunet

Verifier que le container utilise bien Python 3.7 :

```bash
make check
```

Si l'erreur persiste, reconstruire l'image :

```bash
make clean
make start
```

## Stack technique

| Composant | Version | Role |
|-----------|---------|------|
| Python | 3.7 | Runtime |
| TensorFlow | 1.15.5 | Backend MCUNet |
| PyTorch | 1.9.0 (CPU) | Modeles fp32, analyse couches |
| tflite-runtime | 2.7.0 | Inference int8 |
| MCUNet | latest | Model Zoo, architectures |
| TinyEngine | latest | Analyse deploiement MCU |
| nano / vim | - | Edition des scripts dans le container |

## Ressources complementaires

- [MCUNet (GitHub)](https://github.com/mit-han-lab/mcunet) - Repo officiel
- [TinyEngine (GitHub)](https://github.com/mit-han-lab/tinyengine) - Moteur d'inference MCU
- [MCUNet paper (NeurIPS 2020)](https://arxiv.org/abs/2007.10319)
- [MCUNetV2 paper (NeurIPS 2021)](https://arxiv.org/abs/2110.15352)
- [Cours EfficientML du MIT](https://efficientml.ai) - Cours complet sur le ML efficace
- [Tiny Machine Learning Open Education Initiative (TinyMLedu)](https://tinyml.seas.harvard.edu/)
