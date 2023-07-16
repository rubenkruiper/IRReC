# Information Retrieval for Regulatory Compliance (IRReC)

---
Python-based Information Retrieval system over UK building regulations, also see `system_description_slides.pdf`. The
system relies on:
* [SPaR.txt](https://github.com/rubenkruiper/SPaR.txt) for discovering concepts (incl. discontiguous multi-word expressions) in the domain of building regulations.
* [Haystack](https://github.com/deepset-ai/haystack/) as an IR framework, which under the hood relies on:
  * Sparse retrieval is performed through [Elasticsearch](https://www.elastic.co/downloads/elasticsearch) 
  * Dense retrieval is performed using the [FAISS](https://faiss.ai/) library, and DPR encodings.
* For using a Knowledge Graph to (try to) improve results using Query Expansion, load the
graph in [the standard (free) GraphDB docker container](https://github.com/Ontotext-AD/graphdb-docker).

---
**To use the IR system**:
* **Elasticsearch**: should be running on the default port 9200, which will provide sparse indexing and retrieval.
  * To enable localhost HTTP calls on ES 8 and later, disable SSL, e.g., `./elasticsearch -E xpack.security.enabled=false`
* **GraphDB** should be running on on the default port 7200, which enables using the Knowledge Graph / vocabularies 
for Query Expansion. The expected repository name is `BRE`.
* **British Standards**: make sure the input PDF-files are available to the system at the path `datavolume/ir_data/pdf/`.
* **SPaR.txt**: copy the SPaR.txt data to `datavolume/spar_data/`
* **Docker**: make sure docker and docker compose are available.

Clone this repository and cd into it. Prepare the SPaR.txt container:
1. build: `docker compose build spar`
2. run to train a model: `docker-compose up --no-deps spar`
Then simply run `docker compose up` to build and start all containers. We have been running our final tests with the system on a VPS server running Ubuntu 22.04 and 24GB of RAM, but without a GPU. The system should be able to set up and run faster when running with a GPU. 
<!-- ; the assumption is that you have one of the following:
1. One or several pre-trained clustering models, enabling clustering of unseen text on a CPU. To this end copy the contents
  the `cluster_data.zip` file to the path: `datavolume/cluster_data`
2. A GPU-enabled machine, with KMCUDA set up correctly (also see note see below) to be able to create your own clustering model. -->

<!-- **FIRST RUNS**  
When the system is being run for the first time, each of the docker containers is trying
to access each-others API's to continue their own processes. Therefore, it may be required to
re-initialise some of the containers. Options include:
* restart the whole system with `docker compose up`
* restart single container from separate terminal window `sudo docker-compose up --force-recreate --no-deps -d $container_name`
with `$container_name` being one of: `haystack`, `cluster`, `spar`, `query_expansion`. -->


---
**Updating settings before indexing, or during query time**

You can set which type of retriever(s) you'd like to use, and the settings for these, e.g., which fields to index, the type of DPR model to use for dense retrieval, where to store indices, where to cache embeddings, etc. 

* When simply wanting to set up the system's input folder, background corpus, and so on, you'll have to update `datavolume/information_retrieval_settings.json`. The folder `datavolume` is shared between the different containers, and this settings file contains all relevant settings for the different modules.

* For comparing different retrievers, I'd recommend using the default settings and updating the settings using the API (unless you'd want to avoid computationally intensive dense indexing). 


**Figure:** The QueryExpansion API (and swagger docs) allows you to update the parameters of the system:
![alt text](https://github.com/rubenkruiper/irrec/blob/main/swagger_qe.jpeg?raw=true)
![alt text](https://github.com/rubenkruiper/irrec/blob/main/swagger_parameters.jpeg?raw=true)

---
<!-- Old clustering module

If interested in **training a clustering model** from scratch:

* The clustering container relies on [KMCUDA](https://github.com/src-d/kmcuda)
and, therefore, it is important to adjust the `Dockerfile` inside the Clustering folder to ensure
KMCUDA installs correctly for your system setup. An Nvidia CUDA-enabled GPU is required, currently the system runs in 
Ubuntu 18.04 on a Titan Xp (CUDA_ARCH:52, Driver Version: 470.103.01, CUDA Version: 11.4). 
* When using KMCUDA, it becomes possible to rely on cosine-based KMeans. -->

