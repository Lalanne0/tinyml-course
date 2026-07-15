FROM python:3.7-slim

LABEL maintainer="TinyML Course"
LABEL description="Environnement Docker pour le cours TinyML avec MCUNet"

# Eviter les prompts interactifs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# matplotlib sans ecran (container headless) : on ecrit des fichiers PNG
ENV MPLBACKEND=Agg

# Dependances systeme (nano + vim pour que les apprenants editent les scripts)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    unzip \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    nano \
    vim \
    less \
    && rm -rf /var/lib/apt/lists/*

# Versions Python pinees pour compatibilite TensorFlow 1.15
RUN pip install --no-cache-dir \
    gast==0.2.2 \
    tensorflow==1.15.5

RUN pip install --no-cache-dir \
    tflite-runtime==2.7.0

RUN pip install --no-cache-dir \
    numpy==1.19.5 \
    protobuf==3.19.6 \
    h5py==2.10.0

# PyTorch CPU-only
RUN pip install --no-cache-dir \
    torch==1.9.0+cpu \
    torchvision==0.10.0+cpu \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Outils Python supplementaires (plus de Jupyter : le cours est en scripts .py)
RUN pip install --no-cache-dir \
    pillow \
    matplotlib \
    pandas \
    thop

# Cloner MCUNet et installer en mode editable
RUN git clone https://github.com/mit-han-lab/mcunet.git /opt/mcunet \
    && cd /opt/mcunet \
    && pip install --no-cache-dir -e .

# Cloner TinyEngine
RUN git clone https://github.com/mit-han-lab/tinyengine.git /opt/tinyengine

# Creer le repertoire de travail
WORKDIR /workspace

# Telecharger le dataset VWW minival (~380 MB)
# Le tarball s'extrait en dataset/vww-s256/val/{0,1} ou 1=person, 0=non-person
RUN mkdir -p /dataset/vww_minival && \
    wget -q --show-progress -O /tmp/vww-minival.tar \
    "https://www.dropbox.com/s/bc7qi89ezra9711/vww-minival.tar?dl=1" && \
    tar -xf /tmp/vww-minival.tar -C /dataset/vww_minival/ && \
    rm /tmp/vww-minival.tar && \
    ln -s /dataset/vww_minival/dataset/vww-s256/val/1 /dataset/vww_minival/person && \
    ln -s /dataset/vww_minival/dataset/vww-s256/val/0 /dataset/vww_minival/non_person

# Pre-telecharger les modeles MCUNet (cache dans ~/.cache/mcunet/)
RUN printf "from mcunet.model_zoo import build_model, download_tflite\n\
for net_id in ['mcunet-vww0', 'mcunet-vww1', 'mcunet-vww2', 'mcunet-in2', 'mcunet-in3']:\n\
    print(f'Telechargement du modele {net_id}...')\n\
    build_model(net_id=net_id, pretrained=True)\n\
    download_tflite(net_id=net_id)\n\
    print(f'{net_id} OK')\n\
" > /tmp/download_models.py && python /tmp/download_models.py

# Creer des alias pour les modeles TFLite avec des noms lisibles
# MCUNet stocke dans ~/.torch/mcunet/ avec des noms internes (ex: mcunet-5fps_vww)
# Les exercices referencent ~/.mcunet/<net_id>.tflite
RUN mkdir -p /root/.mcunet && \
    ln -s /root/.torch/mcunet/mcunet-10fps_vww.tflite /root/.mcunet/mcunet-vww0.tflite && \
    ln -s /root/.torch/mcunet/mcunet-5fps_vww.tflite /root/.mcunet/mcunet-vww1.tflite && \
    ln -s /root/.torch/mcunet/mcunet-320kb-1mb_vww.tflite /root/.mcunet/mcunet-vww2.tflite && \
    ln -s /root/.torch/mcunet/mcunet-256kb-1mb_imagenet.tflite /root/.mcunet/mcunet-in2.tflite && \
    ln -s /root/.torch/mcunet/mcunet-320kb-1mb_imagenet.tflite /root/.mcunet/mcunet-in3.tflite

# Installer les dependances TinyEngine (flatbuffers pour le parser tflite)
RUN pip install --no-cache-dir flatbuffers==2.0

# Copier les scripts utilitaires, les exercices, les solutions et le guide.
# Note : exercices et solutions sont aussi montes en volume par docker-compose,
# le montage prend le dessus a l'execution (edition persistee sur l'hote).
COPY scripts/ /workspace/scripts/
COPY exercises/ /workspace/exercises/
COPY solutions/ /workspace/solutions/
COPY START_HERE.txt /workspace/START_HERE.txt

# Afficher le guide de demarrage a chaque ouverture d'un shell interactif
RUN echo 'cat /workspace/START_HERE.txt 2>/dev/null || true' >> /root/.bashrc

# Le container reste actif en arriere-plan ; les apprenants s'y connectent
# avec `make shell` (docker compose exec ... bash).
CMD ["sleep", "infinity"]
