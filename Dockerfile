FROM tensorflow/tensorflow:latest-gpu-py3

RUN pip install --upgrade pip
RUN pip install \
        theano==1.0.5 \
        Cython==0.29.21 \
        numpy==1.18.1 \
        matplotlib==3.3.1 \
        seaborn==0.10.1 \
        scipy==1.4.1 \
        scikit-learn==0.23.2 \
        pandas==1.1.0 \
        pyyaml==5.3.1 \
        imutils \
        opencv-python \
        torch==1.6.0 \
        tqdm==4.48.2 \
        psutil==5.7.2 \
        h5py==2.10.0 \
        jupyter \ 
        hdf5storage

WORKDIR /app/codalab
