# ITEM CLASSIFICATION

## Project Overview
This project aims to streamline item classification using advanced NLP and computer vision techniques, making it easier to automate categorization tasks efficiently.
The objective is to conduct a feasibility study for an item classification engine based on textual descriptions and/or images.

## Data Access
The dataset for this project is available at [this address](https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Textimage+DAS+V2/Dataset+projet+pre%CC%81traitement+textes+images.zip).

## Project Objectives
To achieve item classification, the following steps will be undertaken:

1. **Dataset Analysis**
   - Preprocessing of product descriptions and images.
   - Dimensionality reduction.
   - Clustering of items.
   - Visualization of results in two-dimensional graphs.
   - Similarity calculations (e.g., Adjusted Rand Index - ARI) to compare real categories with generated clusters.

2. **Text-Based Classification**
   - Implementing different text feature extraction techniques:
     - Bag-of-Words (simple word counting and TF-IDF).
     - Word/Sentence embedding using Word2Vec.
     - Word/Sentence embedding using BERT.
     - Word/Sentence embedding using Universal Sentence Encoder (USE).
   
3. **Image-Based Classification**
   - Extracting image features using:
     - Traditional algorithms such as SIFT.
     - CNN-based Transfer Learning approaches.

## Project Structure
The project consists of three main notebooks:
- **text-preprocessing**: Handles text data preprocessing.
- **text-clustering-and-classification**: Performs item clustering and classification using text data.
- **image-clustering-and-classification**: Conducts item clustering and classification using images.

## Tools & Technologies
- **NLP Libraries**: NLTK, Word2Vec, BERT, USE
- **Machine Learning**: Scikit-learn, TensorFlow, Hugging Face
- **Computer Vision**: OpenCV, VGG-16
- **Visualization**: Matplotlib, Seaborn


