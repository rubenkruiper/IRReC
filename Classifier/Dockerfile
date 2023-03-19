#FROM nvidia/cuda:11.4.2-devel-ubuntu18.04
FROM python:3.8


RUN apt-get update

# install Python and pip
RUN apt-get install python3-pip -y
RUN python3 -m pip install  pip --upgrade

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
CMD ["uvicorn", "classifier_api:Classifier_api", "--host", "0.0.0.0", "--port", "8502", "--reload"]