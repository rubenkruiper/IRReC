FROM nvidia/cuda:11.4.2-base-ubuntu18.04


RUN apt-get update && \
    apt-get install --no-install-recommends -y wget libfontconfig1 -y

# install pdftotext
RUN wget --no-check-certificate https://dl.xpdfreader.com/xpdf-tools-linux-4.04.tar.gz && \
    tar -xvf xpdf-tools-linux-4.04.tar.gz && cp xpdf-tools-linux-4.04/bin64/pdftotext /usr/local/bin

# install Python, pip and requirements
RUN wget https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN apt-get install python3.8 -y python3-pip
RUN python3.8 -m pip install pip --upgrade
COPY requirements.txt app/
RUN python3.8 -m pip install -r ./app/requirements.txt

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
