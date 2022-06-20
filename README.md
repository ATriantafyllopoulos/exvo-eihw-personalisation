# EXVO

Code to run EXVO experiments for https://arxiv.org/abs/2206.06680.
Code is based on downloading the original data.

First install requirements (preferrably in virtualenv):

```bash
pip install -r requirements.txt
```

Then run `./melspects.py` to extract features, 
followed by `training.py` to run baseline training for Task1 and Task 3
and `./personalisation.py` to run proposed training for Task3.
