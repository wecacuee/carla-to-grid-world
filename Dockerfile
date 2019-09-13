FROM carlasim/carla

USER root
RUN apt-get update \
    && \
    apt-get install -y \
        libpng16-dev \
        libjpeg8-dev \
        libtiff5-dev \
        python3-pip \
    && \
    rm -rf /var/lib/apt/lists/*
ENV PYTHONPATH=/home/carla/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg:$PYTHONPATH
USER carla
