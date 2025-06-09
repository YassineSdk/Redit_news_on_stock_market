import praw
import datetime as dt
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime,timezone,time
import pandas as pd 
import plotly.graph_objects as go
from transformers import  AutoTokenizer , AutoModelForSequenceClassification,pipeline
import streamlit as st
from functions import loading_data,cleaning_data,get_sentiment,map_sentiment,sentiment,loading_yfinance,final_data,sectors
import re
import yfinance as yf
import plotly.express as px
import numpy as np
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title='Signal analysis of stocks',layout='wide')

image_url = "https://i.ibb.co/4g0m58X8/3334896.jpg"

page_bg_img = f"""
<style>
html, body {{
    height: 100%;
    margin: 0;
}}

[data-testid="stAppViewContainer"] {{
    background-image: url("{image_url}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

main .block-container {{
    padding-top: 2rem;
    background-color: rgba(0, 0, 0, 0); /* transparent main container */
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("NLP-Powered Financial Sentiment Tracker")

st.sidebar.header('please insert the informations')
start = st.sidebar.date_input(label = 'From')
end = st.sidebar.date_input(label = 'To',min_value=datetime(2000,1,1),max_value=datetime.today())
field = st.sidebar.selectbox(label='Select a field',options=list(sectors.keys()))
ticker = st.sidebar.selectbox(label='select the stock',options=sectors[field])
button = st.sidebar.button(label='analyse')
if not button :
    st.markdown("""
        ### 📊 Stock Signal Analyzer: Reddit Sentiment Meets Market Data

        This interactive dashboard combines **financial data** with **natural language processing (NLP)** to analyze how public sentiment on Reddit affects stock prices.

        ✅ Pulls Reddit posts mentioning selected stocks  
        ✅ Cleans and processes textual data using FinBERT (a finance-specific transformer model)  
        ✅ Visualizes sentiment signals alongside historical stock prices  
        ✅ Helps identify potential market signals driven by retail investor sentiment  

        Select a stock and a date range from the sidebar to start the analysis.
        """)

if button :
    if end < start :
        st.error('the end date must be greater than the start date')
    else :    
    # connverting the dates 
        progress_bar = st.progress(0, text="Starting analysis...")
        start_datetime = datetime.combine(start, time.min).replace(tzinfo=timezone.utc)
        end_datetime = datetime.combine(end, time.max).replace(tzinfo=timezone.utc)
        start_timestamp = int(start_datetime.timestamp())
        end_timestamp = int(end_datetime.timestamp())
        data = loading_data(start_datetime,end_datetime,ticker)
        progress_bar.progress(10, text="Data preprocessing the data ")
    #loading the data     
        progress_bar.progress(25, text="Data imported")
    #cleaning the data     
        data_clean = cleaning_data(data)    
        progress_bar.progress(50, text="Data is cleaned")
    # sentiment analysis    
        with st.spinner('Loading FinBERT model and running sentiment analysis...'):
            model_name = "ProsusAI/finbert"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            nlp = pipeline('sentiment-analysis',model=model,tokenizer=tokenizer,truncation=True)
            data[['sentiment','sentiment_score']] = data['self_text'].apply(lambda x : get_sentiment(x,nlp)).apply(pd.Series)
            raw_score = data['score'] * 0.5 * data['sentiment_score'] * data['sentiment'].map(map_sentiment)
            data['reinforced_sentiment'] = np.tanh(raw_score)
        progress_bar.progress(75, text="Sentiment analysis is Done")        
        df = loading_yfinance(ticker,start,end)
        final_data = final_data(df,data)

        color_map = {
            'positive': 'green',
            'neutral': 'gray',
            'negative': 'red'
        }
        # Map the colors
        final_data['color'] = final_data['sentiment'].map(color_map)

        #plotting the sentiment :
        fig = go.Figure()

        # Closing Price (right y-axis #1)
        fig.add_trace(go.Scatter(
            x=final_data['date'],
            y=final_data['Close'],
            mode='lines+markers',
            yaxis='y2',
            name='Closing price on the NVDA stocks',
            line=dict(color='green')
        ))

        # Reinforced Sentiment (left y-axis)
        fig.add_trace(go.Scatter(
            x=final_data['date'],
            y=final_data['reinforced_sentiment'],
            mode='lines+markers',
            yaxis='y1',
            name='Sentiment',
            marker=dict(color=final_data['color']),
            line=dict(color='blue')
        ))



        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title='Reinforced Sentiment and Closing Price evolution',
            yaxis=dict(
                title='Reinforced Sentiment',
                side='left',
                showgrid=False  
            ),
            
            yaxis2=dict(
                title='Closing Price',

                anchor='x',
                overlaying='y',
                side='right',
                position=0.85,
                showgrid=False 
            ),
            
            legend=dict(x=0.1, y=1.1, orientation='h'),
            margin=dict(l=60, r=120, t=80)
        )


        st.plotly_chart(fig, use_container_width=True)
        # sentiment distribution :
        sentiment_count =data['sentiment'].value_counts().reset_index()
        sentiment_count.columns = ['sentiment','count']
        fig_1 = px.bar(sentiment_count,
                    x='sentiment',
                    y='count',
                    title='sentiment distribution')
        
        fig_1.update_layout(
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
            
        )            

        # sentiment by subreddit 
        subreddit_sentiment_count = (
        data.groupby(['subreddit', 'sentiment'])
        .size()
        .reset_index(name='count')
        )

        fig_subreddit_sentiment = px.bar(
            subreddit_sentiment_count,
            x='subreddit',
            y='count',
            color='sentiment',
            title='Sentiment Distribution per Subreddit',
            barmode='stack',  
            category_orders={"sentiment": ["positive", "neutral", "negative"]} 
        )
        fig_subreddit_sentiment.update_layout(
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        progress_bar.progress(100, text="Analysis complete!")
        col_1 , col_2 = st.columns(2)
        with col_1 :
            st.plotly_chart(fig_1, use_container_width=True)
        with col_2 :
            st.plotly_chart(fig_subreddit_sentiment, use_container_width=True,background=False)














































