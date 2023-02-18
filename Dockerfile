FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

COPY environment.yml /tmp/environment.yml

# Create environment with base packages
RUN conda create --name Dissertation --clone base && \
    conda clean --all --yes --force-pkgs-dirs && \
    rm -rf /opt/conda/pkgs/*

# Update environment with packages from environment.yml
RUN conda env update --name Dissertation --file /tmp/environment.yml && \
    conda clean --all --yes --force-pkgs-dirs && \
    rm -rf /opt/conda/pkgs/*

# Set working directory and default command
COPY . /app
WORKDIR /app
CMD ["python", "main.py"]
