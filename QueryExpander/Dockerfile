FROM python:3.8

COPY requirements.txt app/
RUN python3 -m pip install -r ./app/requirements.txt

# install app
ARG ONLY_CODE=unkown
RUN echo "$ONLY_CODE"
COPY . app/
WORKDIR /app

EXPOSE 8503
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# CMD ["uvicorn", "qe_api:QE_api", "--root-path", "/brretrieval", "--host", "0.0.0.0", "--port", "8503", "--reload"]
CMD ["uvicorn", "qe_api:QE_api", "--host", "0.0.0.0", "--port", "8503", "--reload"]


