# ITEM-CLASSIFICATION

The data are available at this address : https://drive.google.com/drive/folders/1WVf2xV9KDRm4TDKFcs6Ti1EMO9-mtCBR?usp=sharing

**Project Objectives :**
The main goal of this project is to conduct the feasibility study of an item classification engine, based on textual description and/or images, for automating item categorization. To do this, we are going to :
- Analyze the dataset by preprocessing product descriptions and images, reducing dimensionality, and performing clustering. The results of dimensionality reduction and clustering will be presented in two-dimensional graphs and confirmed by similarity calculation (e.g., ARI) between real categories and clusters.
- Train Natural Language Processing (NLP) and computer vision models to classify items.

To extract textual features, we will implement several approaches :
- Two "bag-of-words" approaches: simple word counting and TF-IDF.
- A traditional word/sentence embedding approach using Word2Vec.
- A word/sentence embedding approach using BERT.
- A word/sentence embedding approach using USE (Universal Sentence Encoder).

To extract image features, we will use two strategies :
- An algorithm such as SIFT / ORB / SURF.
- A CNN Transfer Learning-based algorithm.

The **text-preprocessing** notebook is dedicated to text data preprocessing, the **text-clustering-and-classification** notebook to item clustering and classification using text data, and the **image-clustering-and-classification** notebook to item clustering and classification using images.