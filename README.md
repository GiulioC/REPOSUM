# Project REPOSUM
This repository was created to store the code for the main solutions and applications developed during my 1-year research grant working at the project REPOSUM for Università degli Studi di Torino.

See also:
- [REPOSUM-classification](https://github.com/GiulioC/REPOSUM-Classification)
- [pyRelFinder](https://github.com/GiulioC/pyRelFinder)
- [distant-reading-UniTO](https://github.com/GiulioC/distant-reading-UniTO)

Below a brief description of the content of this repo. For more information about the single projects, read the <b>final report</b> of the research activity at the following [link](https://giulioc.github.io/files/Relazione_Finale_REPOSUM_Carducci.pdf) (only available in italian).

Careers
-----
An attempt at inferring the academic success of philophers after the completion of their PhD. The system leverage a linear combination of features extracted from data that had been manually labeled by domain experts.


Classification
-----
Identification of philosophical theses among a dataset of about 500K unlabeled documents, using the semantic features from their title. Feature extraction is carried out with a BoW approach and the classifier is Random Forest. Classification results are then enhanced using a semantic inference approach based on the knowledge base babelnet.


Classification with TellMeFirst
-----
A different classification approach that leverages entities extracted from [TellMeFirst](https://tellmefirst.synapta.io/) as features. The semantic inference step is replaced by dimensionality reduction using SVD. 


Entity Recognition
-----
A simple implementation of entity recognition in a jupyter notebook using ```spacy```.


Semantic Inference
-----
Definition of a knowledge graph built with entities extracted from the documents using tellmefirst and philosophical entities extracted from Wikidata. The KG can the be used for semantic inference and reasoning using graph mining techniques, e.g. shortest path and betweennes centrality.


Topic Modeling
-----
Computation of a fixed number of topics from unstructured textual content of the document, using a unsupervised clustering technique.
