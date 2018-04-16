# Deep Text Classifier

Implementation of document classification model described in [Hierarchical Attention Networks for Document Classification (Yang et al., 2016)](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf).

## How to run

1. Create a virtual environment, activate it, and install requirements:
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

2. Download the English model for spaCy:

```
python -m spacy download en
```

3. Get [Yelp review dataset](https://www.yelp.com/dataset_challenge) and extract it in this directory.
```
python3 yelp_prepare.py dataset/review.json
python3 worker.py --mode=train --device=/gpu:0 --batch-size=30
```

## Results
I am getting 65% accuracy on a dev set (16% of data) after 3 epochs. Results reported in the paper are 71% on Yelp'15.
No systemic hyperparameter optimization was performed.