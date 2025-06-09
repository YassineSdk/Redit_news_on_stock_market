import praw
import datetime as dt
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime,timezone
import pandas as pd 
import plotly.graph_objects as go
from transformers import  AutoTokenizer , AutoModelForSequenceClassification,pipeline
import re
import numpy as np
import yfinance as yf



def loading_data(start_datetime, end_datetime, ticker):
    reddit = praw.Reddit(
        client_id="eK7Gid3MOrvEWkCkXqHpVw",
        client_secret="vRj93NZQIDXpR43cX07CeESEcNsg7w",
        user_agent="SentimentBot"
    )

    subreddits = ["wallstreetbets", "stocks", "investing", "options", "ValueInvesting", "Finance", "pennystocks"]
    reddit_posts = []

    for subreddit in subreddits:
        for submission in reddit.subreddit(subreddit).search(ticker, limit=1000):
            post_time = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)

            if start_datetime <= post_time <= end_datetime:
                reddit_posts.append({
                    'title': submission.title,
                    'self_text': submission.selftext,
                    'date': post_time.strftime('%Y-%m-%d'),
                    'score': submission.score,
                    'subreddit': str(submission.subreddit),                   
                })

    data = pd.DataFrame(reddit_posts)
    return data




def clean_text(text):
    # Remove markdown links 
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    # Remove stray markdown artifacts 
    text = re.sub(r'\[\]|\(\)', '', text)
    # Replace \xa0 (non-breaking space) and \n (newlines) with space
    text = text.replace('\xa0', ' ').replace('\n', ' ')
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) 
    # strip leading/trailing whitespace
    return text.strip()



def cleaning_data(data):
    data['date'] = pd.to_datetime(data['date'])
    data.sort_values(by='date',ascending=True,inplace=True)
    data.sort_index(inplace=True)
    data['self_text'] =  data['self_text'].apply(lambda x : clean_text(x))
    data['self_text'].apply(lambda x:  np.nan if x.strip() == '' else x)
    data.dropna(axis=1,inplace=True)
    data['score'] = data['score'].astype(int)
    data.set_index('date',inplace=True)
    data.sort_index(inplace=True)   
    return data  





def get_sentiment(text,nlp):
    result = nlp(text)[0]
    return result['label'],result['score']




def loading_yfinance(ticker,start,end):
    df = yf.download(ticker,start=start,end=end)
    df.columns =df.columns.droplevel(1)
    df['date'] = df.index
    return df 


def map_sentiment(sentiment):
    if sentiment == "positive" :
        return 1
    elif sentiment == "negative":
        return -1
    elif sentiment == "neutral":
        return 0 

def sentiment(reinforced_sentiment):
    if reinforced_sentiment > 0 :
        return "positive"
    elif reinforced_sentiment == 0 :
        return   "neutral"
    elif  reinforced_sentiment < 0 :
        return "negative"

def  final_data(df,data):
    sentiment_evolution = data.groupby(data.index)['reinforced_sentiment'].mean().reset_index() 
    df_1 = pd.merge(sentiment_evolution,df[['Close','date']],on='date',how='left')
    df_1['sentiment'] = df_1['reinforced_sentiment'].map(sentiment)
    df_1.dropna(axis=0,inplace=True)
    return df_1

sectors = {
    "Tech & AI": ["NVDA", "MSFT", "AAPL", "GOOGL", "AMD", "TSLA"],
    "Finance & Crypto": ["JPM", "GS", "COIN", "SQ", "PYPL"],
    "Consumer & Retail": ["AMZN", "WMT", "COST", "NKE"],
    "Entertainment & Gaming": ["NFLX", "DIS", "ATVI", "ROBLX"],
    "Energy & EV": ["XOM", "CVX", "RIVN", "LCID"],
    "Meme Stocks": ["GME", "AMC", "BBBYQ", "PLTR"]
}      












