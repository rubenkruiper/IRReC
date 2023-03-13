# SPaR.txt - a cheap Shallow Parsing approach for Regulatory texts
_[This work was published](https://aclanthology.org/2021.nllp-1.14) in the EMNLP 2021 workshop on [Natural Legal Language Processing](http://nllpw.org/)_    

> **Abstract**: Automated Compliance Checking (ACC) systems aim to semantically parse building regulations to a set of rules. 
However, semantic parsing is known to be hard and requires large amounts of training data. 
The complexity of creating such training data has led to research that focuses on small sub-tasks, 
such as shallow parsing or the extraction of a limited subset of rules. This study introduces a shallow parsing 
task for which training data is relatively cheap to create, with the aim of learning a lexicon for ACC. 
We annotate a small domain-specific dataset of 200 sentences, SPaR.txt, 
and train a sequence tagger that achieves 79,93 F1-score on the test set. We then show through 
manual evaluation that the model identifies most (89,84\%) defined terms in a set of building regulation 
documents, and that both contiguous and discontiguous Multi-Word Expressions (MWE) are discovered with 
reasonable accuracy (70,3\%).

This repository contains:
* ScotReg - a json corpus of the [domestic](https://www.gov.scot/publications/building-standards-technical-handbook-2020-domestic/) and [non-domestic](https://www.gov.scot/publications/building-standards-technical-handbook-2020-non-domestic/) Scottish Building Regulations 
   * scraped 14th of June 2021 
   * please also see [crown copyright](https://www.gov.scot/crown-copyright/) and the [Open Government License](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/)
* 200 randomly selected sentences annotated using [BRAT](https://brat.nlplab.org/), see [the SpaR.txt paper](https://aclanthology.org/2021.nllp-1.14) for annotation details ([arXiv link](https://arxiv.org/abs/2110.01295))
* Code to train a biLSTM+CRF tagger, in support of learning a lexicon (including MWE discovery/identification)


---

**Using SPaR.txt**
Clone the repository and enter the directory from terminal/console.
1. Create a new conda environment, e.g.:
    ```
    conda create -n spar python=3.8
    ```
2. Activate your new environment
    ```
    conda activate spar
    ```
3. Install the dependencies (make sure you are inside the directory)
    ```
    pip install -r requirements.txt
    ```
4. Train a model
    ```
    python run_tagger.py
    ```
5. Use the trained model to:
  * Evaluate the model on the test set
      ```
      python run_tagger.py --evaluate -i "data/test/" --batchsize 8
      ```
  * Predict tags for all the sentences found in ScotReg
      ```
      python run_tagger.py --predict -m "trained_models/debugger_train/" -i "data/all_non_annotated_sents/" -o "predictions/all_sentence_predictions.json" --batchsize 8
      ```
  * Try parsing your own sentences from the terminal (see example below)
      ```
      python serve_spar.py
      ```
    
![Example of input and output from terminal](https://github.com/rubenkruiper/SPaR.txt/ExampleServingSpar.jpg?raw=true)

---

If you use code or data from SPaR.txt in your research, please consider citing the following paper:
```
@inproceedings{kruiper-etal-2021-spar,
    title = "{SP}a{R}.txt, a Cheap Shallow Parsing Approach for Regulatory Texts",
    author = "Kruiper, Ruben  and
      Konstas, Ioannis  and
      Gray, Alasdair J.G.  and
      Sadeghineko, Farhad  and
      Watson, Richard  and
      Kumar, Bimal",
    booktitle = "Proceedings of the Natural Legal Language Processing Workshop 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.nllp-1.14",
    pages = "129--143",
    abstract = "Automated Compliance Checking (ACC) systems aim to semantically parse building regulations to a set of rules. However, semantic parsing is known to be hard and requires large amounts of training data. The complexity of creating such training data has led to research that focuses on small sub-tasks, such as shallow parsing or the extraction of a limited subset of rules. This study introduces a shallow parsing task for which training data is relatively cheap to create, with the aim of learning a lexicon for ACC. We annotate a small domain-specific dataset of 200 sentences, SPaR.txt, and train a sequence tagger that achieves 79,93 F1-score on the test set. We then show through manual evaluation that the model identifies most (89,84{\%}) defined terms in a set of building regulation documents, and that both contiguous and discontiguous Multi-Word Expressions (MWE) are discovered with reasonable accuracy (70,3{\%}).",
}
```
The code and BRAT annotations in this repository are licensed under a Creative Commons Attribution 4.0 License.

<img src="https://mirrors.creativecommons.org/presskit/buttons/88x31/png/by-sa.png" width="134" height="47">
