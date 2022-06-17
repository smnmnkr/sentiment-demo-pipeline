# Frozen BERT Encoder + MultiLayerPerceptron
Warning: The following make calls are only available with OSX/LINUX. If you use Windows, please excuse the inconvenience.

## Install
```bash
### to install and download the spacy/textblob pipelines use:
make install

### manually install:
pip install -r requirements.txt
```

## Usage
```bash
### Predefined Targets:
make run

### Python module
# python3 -m $classifier -C CONFIGS
# e.g.:
python3 -m $classifier -C ./config.json
```

## Results
```
[--- EVAL -> ./data/tweets.eval.csv ---]
AVG           	 tp:     2359	 fp:      569 	 tn:     4718	 fn:      569	 pre=0.8057	 rec=0.8057	 f1=0.8057	 acc=0.8615
neutral       	 tp:      369	 fp:      178 	 tn:     1990	 fn:      264	 pre=0.6746	 rec=0.5829	 f1=0.6254	 acc=0.8422
negative      	 tp:     1663	 fp:      306 	 tn:      696	 fn:      158	 pre=0.8446	 rec=0.9132	 f1=0.8776	 acc=0.8356
positive      	 tp:      327	 fp:       85 	 tn:     2032	 fn:      147	 pre=0.7937	 rec=0.6899	 f1=0.7381	 acc=0.9105
```