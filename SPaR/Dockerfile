#FROM nvidia/cuda:11.4.2-base-ubuntu18.04
FROM python:3.8

RUN apt-get update

# install Python, pip and requirements
# RUN apt-get install python3.8 -y  python3-pip build-essential libssl-dev libffi-dev python3.8-dev
RUN apt-get install python3-pip -y build-essential libssl-dev libffi-dev python3.8-dev
RUN python3.8 -m pip install  pip --upgrade
COPY requirements.txt app/
RUN python3.8 -m pip install -r ./app/requirements.txt
#RUN ["python3.8", "-c", "import nltk; nltk.download('punkt', download_dir='/usr/local/nltk_data'); nltk.download('omw-1.4', download_dir='/usr/local/nltk_data')"]
RUN [ "python3.8", "-c", "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('omw-1.4')" ]
RUN cp -r /root/nltk_data /usr/local/share/nltk_data

# install app
ARG ONLY_CODE=unkown
RUN echo "$ONLY_CODE"
COPY . app/
WORKDIR /app

EXPOSE 8501
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

CMD ["uvicorn", "spar_api:SPaR_api", "--host", "0.0.0.0", "--port", "8501", "--reload"]
