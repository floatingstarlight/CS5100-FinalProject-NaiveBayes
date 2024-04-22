# CS5100-FinalProject-NaiveBayes
NaiveBayes Related code

This is the code for the final project of CS5100.
In our project, we try to do sentiment analysis for the dataset of reddit & twitter comments.
Specifically, this repo contains code for the Naive Bayes Model & its related experiments/variants/comparisions.

**1.Manual Implementation of Naïve Bayes Model:** the driven/main code is in the file ```naiveNB_base.py```. Run this file to test for manual implementation of Naive Bayes model.

```Text-preprocessor.py``` contains the TextPreprocessor class that deals with all preprocessing. This is a helper class. 

```NaiveBayeClassifer.py``` contains the a class that performs the major algorithm of Naive Bayes. This is a helper class. 

In addition, some variations are experiments on the manual implementation of NB model. This is also included in the ```naiveNB_base.py```.
- Use unigram vs. Bigram during feature selection
- Preprocessing variants:
   - Unigram + No preprocessing 
   - Unigram + Replace Abbrevations Words
   - Unigram + Remove stop words
   - Unigram + Lemmatization
   - Unigram + StemmingWords Remove

**2.Built-in library:** MultinomialNB in sklearn library, code in ```multimonimalNB.py```

**3.Built-in library:** Linear SVC, code in ```LinearSVC.py```

