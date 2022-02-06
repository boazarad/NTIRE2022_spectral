FROM tensorflow/tensorflow:latest-gpu-py3

ENV TZ=Asia/Jerusalem
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezonei

RUN apt-get update && apt-get install -y python3-opencv

RUN pip install --upgrade pip
RUN pip install \
        numpy==1.18.1 \
        matplotlib==3.3.1 \
        scipy==1.4.1 \
        scikit-learn==0.23.2 \
        pandas==1.1.0 \
        pyyaml==5.3.1 \
        imutils \
        opencv-python \
        tqdm==4.48.2 \
        psutil==5.7.2 \
        h5py==2.10.0 \
        hdf5storage

WORKDIR /app/codalab
