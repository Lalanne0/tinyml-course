# TinyML avec MCUNet

Cas d'etude pratique pour deployer des modeles de deep learning sur microcontroleurs avec le framework MCUNet du MIT.

**Tache fil rouge :** Visual Wake Words (VWW) - classification binaire "personne presente / absente" sur des images 80x80, ciblee sur un STM32F746.

## Prerequis

- **Docker Desktop** (Mac/Windows) ou **Docker Engine** (Linux)
- **8 GB de RAM** libre minimum
- **10 GB de disque** libre (image Docker + dataset VWW)

> **Apple Silicon (M1/M2/M3) :** TensorFlow 1.15 n'a pas de wheel ARM64 native. Ajoutez `--platform=linux/amd64` dans le `docker-compose.yml` (deja configure). L'execution sera plus lente via emulation.

## Demarrage rapide

```bash
git clone <url-du-repo>
cd tinyml-course
make start
```

Ouvrez ensuite [http://localhost:8888](http://localhost:8888) dans votre navigateur.

## Plan des notebooks

| # | Notebook | Objectif | Duree estimee |
|---|----------|----------|---------------|
| 00 | `00_setup_check` | Verifier l'environnement | 10 min |
| 01 | `01_explore_model_zoo` | Explorer le Model Zoo MCUNet | 30 min |
| 02 | `02_inference_python` | Inference TFLite int8 avec quantification | 45 min |
| 03 | `03_compare_vs_mobilenet` | Comparer MCUNet vs MobileNetV2 | 45 min |
| 04 | `04_deployment_check` | Analyse de deploiement avec TinyEngine | 40 min |
| 05 | `05_what_if_it_doesnt_fit` | Strategies quand le modele ne tient pas | 30 min |

**Duree totale estimee :** environ 3h30.

## Architecture du repo

```
tinyml-course/
|-- docker/
|   |-- Dockerfile             # Image Python 3.7 + TF 1.15 + MCUNet
|   +-- docker-compose.yml     # Configuration du service
|-- notebooks/                  # Notebooks a completer par l'apprenant
|   |-- 00_setup_check.ipynb
|   |-- 01_explore_model_zoo.ipynb
|   |-- 02_inference_python.ipynb
|   |-- 03_compare_vs_mobilenet.ipynb
|   |-- 04_deployment_check.ipynb
|   +-- 05_what_if_it_doesnt_fit.ipynb
|-- solutions/                  # Notebooks avec solutions completes (lecture seule)
|-- scripts/
|   +-- tinyengine_analysis.py  # Wrapper Python pour TinyEngine
|-- data/                       # Peuple par Docker (non versionne)
|-- Makefile                    # Commandes : start, stop, check, clean, logs
+-- README.md
```

## Commandes disponibles

| Commande | Description |
|----------|-------------|
| `make start` | Demarre le container et affiche l'URL Jupyter |
| `make stop` | Arrete le container |
| `make check` | Execute le notebook 00 pour verifier l'environnement |
| `make clean` | Supprime container, image et volumes |
| `make logs` | Affiche les logs du container |

## Troubleshooting

### Docker manque de memoire

Augmentez la RAM allouee a 8 GB minimum dans Docker Desktop > Settings > Resources.

### Port 8888 deja utilise

Modifiez le port dans `docker/docker-compose.yml` :

```yaml
ports:
  - "9999:8888"  # remplacez 9999 par un port libre
```

### Build trop long

C'est normal au premier lancement (~15-20 min). Le dataset VWW fait 380 MB et les modeles sont pre-telecharges. Les builds suivants seront quasi instantanes grace au cache Docker.

### Erreur d'import mcunet

Verifiez que le container utilise bien Python 3.7 :

```bash
make check
```

Si l'erreur persiste, reconstruisez l'image :

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
| JupyterLab | latest | Interface pedagogique |

## Ressources complementaires

- [MCUNet (GitHub)](https://github.com/mit-han-lab/mcunet) - Repo officiel
- [TinyEngine (GitHub)](https://github.com/mit-han-lab/tinyengine) - Moteur d'inference MCU
- [MCUNet paper (NeurIPS 2020)](https://arxiv.org/abs/2007.10319)
- [MCUNetV2 paper (NeurIPS 2021)](https://arxiv.org/abs/2110.15352)
- [Cours EfficientML du MIT](https://efficientml.ai) - Cours complet sur le ML efficace
- [Tiny Machine Learning Open Education Initiative (TinyMLedu)](https://tinyml.seas.harvard.edu/)
