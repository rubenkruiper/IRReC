# Information Retrieval for Regulatory Compliance (IRReC)

---
Python-based Information Retrieval system over UK building regulations, also see `system_description_slides.pdf`. The
system relies on:
* [SPaR.txt](https://github.com/rubenkruiper/SPaR.txt) for multi-word concept identification in the domain of building
regulations.
* [Haystack](https://github.com/deepset-ai/haystack/) as an IR framework, which under the hood relies on:
  * Sparse retrieval is performed through [Elasticsearch](https://www.elastic.co/downloads/elasticsearch) 
  * Dense retrieval is performed using the [FAISS](https://faiss.ai/) library.
* For using a Knowledge Graph to (try to) improve results using Query Expansion, load the
graph in [the standard (free) GraphDB docker container](https://github.com/Ontotext-AD/graphdb-docker).

---
**To use the IR system**:
* **Elasticsearch**: should be running on the default port 9200, which will provide sparse indexing and retrieval.
* **GraphDB** should be running on on the default port 7200, which enables using the Knowledge Graph / vocabularies 
for Query Expansion. The expected repository name is `BRE`.
* **British Standards**: make sure the PDF-files are available to the system at the path `datavolume/ir_data/pdf/`.
* **SPaR.txt**: copy the SPaR.txt data to `datavolume/spar_data/`
* **Docker**: make sure docker and docker compose are available.

Clone this repository and simply run `docker compose up`; the assumption is that you have one of the following:
1. One or several pre-trained clustering models, enabling clustering of unseen text on a CPU. To this end copy the contents
  the `cluster_data.zip` file to the path: `datavolume/cluster_data`
2. A GPU-enabled machine, with KMCUDA set up correctly (also see note see below) to be able to create your own clustering model.

**FIRST RUNS**  
When the system is being run for the first time, each of the docker containers is trying
to access each-others API's to continue their own processes. Therefore, it may be required to
re-initialise some of the containers. Options include:
* restart the whole system with `docker compose up`
* restart single container from separate terminal window `sudo docker-compose up --force-recreate --no-deps -d $container_name`
with `$container_name` being one of: `haystack`, `cluster`, `spar`, `query_expansion`.


---
**Updating settings before indexing, or during query time**

The folder `datavolume` is shared between the different containers. This shared data volume contains settings files to 
adjust the settings of the system:
  * **retrieval_settings** - here you can set which type of retrievers you'd like to use, and the settings for these, e.g.;
    * which fields to index, the type of DPR model to use for dense retrieval, where to store indices, where to cache embeddings, etc. 
  * **cluster_settings** - here you can adjust the settings of the clustering node, e.g.;
    * settings for the clustering algorithm (kNN metric, number of clusters etc.), select a BERT-based embedding model
  to use, provide files with terms to cluster, as well as a background corpus 
  * **query_expansion_settings** - here you can update the weights for the different indexed fields during query time.
  Note: the API for querying is hosted at port 8503 (default), and provides a handle for updating
  the query_expansion_settings.

---


If interested in **training a clustering model** from scratch:

* The clustering container relies on [KMCUDA](https://github.com/src-d/kmcuda)
and, therefore, it is important to adjust the `Dockerfile` inside the Clustering folder to ensure
KMCUDA installs correctly for your system setup. An Nvidia CUDA-enabled GPU is required, currently the system runs in 
Ubuntu 18.04 on a Titan Xp (CUDA_ARCH:52, Driver Version: 470.103.01, CUDA Version: 11.4). 
* When using KMCUDA, it becomes possible to rely on cosine-based KMeans.

