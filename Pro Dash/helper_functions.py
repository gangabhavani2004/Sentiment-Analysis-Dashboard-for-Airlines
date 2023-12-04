import numpy as np
import pandas as pd

# for scraping tweets
import snscrape.modules.twitter as sntwitter

# for loading the model and tokenizer
from tensorflow.keras.models import load_model
import pickle

# for text processing
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
nltk.download(
    ["punkt", "wordnet", "omw-1.4", "averaged_perceptron_tagger", "universal_tagset"]
)
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import CountVectorizer

# for visualization
import plotly.express as px
import plotly.io as pio
import matplotlib as mpl
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image