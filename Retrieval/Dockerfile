FROM python:3.8
#FROM nvidia/cuda:11.4.2-base-ubuntu18.04

RUN apt-get update && \
    apt-get install --no-install-recommends -y wget libfontconfig1 -y

# install pdftotext
RUN wget --no-check-certificate https://dl.xpdfreader.com/xpdf-tools-linux-4.04.tar.gz && \
    tar -xvf xpdf-tools-linux-4.04.tar.gz && cp xpdf-tools-linux-4.04/bin64/pdftotext /usr/local/bin

# install Python, pip and requirements
RUN wget https://bootstrap.pypa.io/get-pip.py -o get-pip.py
# RUN apt-get install python3.8 -y python3-pip
RUN apt-get install python3-pip  -y
RUN python3.8 -m pip install pip --upgrade
COPY requirements.txt app/
RUN python3.8 -m pip install -r ./app/requirements.txt

## Install miniconda
#ENV CONDA_DIR /opt/conda
#RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
#     /bin/bash ~/miniconda.sh -b -p /opt/conda
## Put conda in path so we can use conda activate
#ENV PATH=$CONDA_DIR/bin:$PATH
#
## install faiss cpu through conda
#RUN conda install -c conda-forge faiss-cpu

# RUN git clone https://github.com/deepset-ai/haystack.git && pip install -e haystack/.[sql,only-faiss-gpu,preprocessing]

# install app
ARG ONLY_CODE=unkown
RUN echo "$ONLY_CODE"
COPY . app/
WORKDIR /app

EXPOSE 8500
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

CMD ["uvicorn", "retrieval_api:Retrieval_api", "--host", "0.0.0.0", "--port", "8500", "--reload"]
