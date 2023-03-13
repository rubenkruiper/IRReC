FROM nvidia/cuda:11.4.2-devel-ubuntu18.04
RUN apt-get update

# install Python and pip
RUN apt-get install python3-pip -y
RUN python3 -m pip install  pip --upgrade

# install textblob, dependencies and pdftotext
RUN python3 -m pip install textblob==0.15.3
RUN [ "python3", "-c", "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('omw-1.4')" ]
RUN apt-get install --no-install-recommends -y wget libfontconfig1 -y
RUN wget --no-check-certificate https://dl.xpdfreader.com/xpdf-tools-linux-4.04.tar.gz && \
    tar -xvf xpdf-tools-linux-4.04.tar.gz && cp xpdf-tools-linux-4.04/bin64/pdftotext /usr/local/bin

# install libkmcuda (versions may change... this really is just a PoC)
RUN python3 -m pip install numpy
RUN apt install -y git build-essential libprotobuf-dev protobuf-compiler software-properties-common
RUN apt install cmake gcc-6 g++-6 -y
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 60 --slave /usr/bin/g++ g++ /usr/bin/g++-6
RUN git clone https://github.com/src-d/kmcuda
RUN cd /kmcuda/src/ && \
    cmake -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.4 -DCMAKE_BUILD_TYPE=Release . && make && \
    CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda CUDA_ARCH=52 pip install libKMCUDA

# install rest of requirements
COPY requirements.txt app/
RUN python3 -m pip install -r ./app/requirements.txt

# finally boot the clustering app
ARG ONLY_CODE=unkown
RUN echo "$ONLY_CODE"
COPY . app/
WORKDIR /app
EXPOSE 8502
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
# uvicorn cluster_api:Cluster_api --host 0.0.0.0 --port 8505 --reload
CMD ["uvicorn", "cluster_api:Cluster_api", "--host", "0.0.0.0", "--port", "8502", "--reload"]