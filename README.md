# Information Retrieval for Regulatory Compliance (IRReC)

---
The Information Retrieval for Regulatory Compliance (IRReC) provides an experimental search engine to search over building regulations. Due to licensing restrictions you will have to provide your own regulations in .pdf format. This repository contains code and some of the data for the IRReC system, published at [EG-ICE 2023](https://www.ucl.ac.uk/bartlett/construction/research/virtual-research-centres/institute-digital-innovation-built-environment/30th-eg-ice). The paper can be found at (link-to-be-added)[].

The IR system and experiments are all written in python. Instructions for preparing the system, running experiments and running a local front end to call the system API are provided in each of the corresponding folders. 

We hope that our data and code will help you research Information Retrieval in the domain of Architecture, Engineering and Construction. Note that the code in this project for Named Entity Recognition (NER) and automatically creating a span-based Knowledge Graph (KG) has been published separately as well, see:
* [SPaR.txt shallower parser for NER](https://github.com/rubenkruiper/SPaR.txt)
* [iReC KG from building regulations](https://github.com/rubenkruiper/irec)

If you use our work in one of your projects, please consider referencing our paper:

```
@inproceedings{Kruiper2023_IRReC,
   title = "Document and Query Expansion for Information Retrieval on Building Regulations",
    author = "Kruiper, Ruben  and
      Konstas, Ioannis  and
      Gray, Alasdair J.G.  and
      Sadeghineko, Farhad  and
      Watson, Richard  and
      Kumar, Bimal",
    month = jul,
    year = "2023",
    keywords= "Information Retrieval,Building Regulations,Query Expansion,Document Expansion",
}
``` 

The code and data in this repository are licensed under a Creative Commons Attribution 4.0 License.
<img src="https://mirrors.creativecommons.org/presskit/buttons/88x31/png/by-sa.png" width="134" height="47">