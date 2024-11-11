import os
from dotenv import load_dotenv

from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate
import openai

import requests
from bs4 import BeautifulSoup
import json

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
bing_api_key = os.getenv("BING_API_KEY")
chat_model = os.getenv("CHAT_MODEL")

def search(query):
    search_url = "https://api.bing.microsoft.com/v7.0/search"
    headers = {
        "Ocp-Apim-Subscription-Key": bing_api_key
    }
    params = {
        "q": query,
        "textDecorations": True, 
        "textFormat": "HTML"
    }
    try:
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
        search_results = response.json()
    except Exception as ex:
        raise ex
    # limit web page 
    pages = search_results["webPages"]["value"]
    n_web = min(10, len(pages))
    search_results["webPages"]["value"] = pages[:n_web]
    return search_results

def scrape(url: str):
    # scrape website, and also will summarize the content based on objective if the content is too large
    # objective is the original objective & task that user give to the agent, url is the url of the website to be scraped
    print("Scraping website...")
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract the text from the parsed HTML
        text = soup.get_text(separator=' ', strip=True)
        
        if len(text) > 4000:
            output = summary(text)
            return output
        else:
            return text
                
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

def summary(content):
    llm = ChatOpenAI(temperature=0, model=chat_model)
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a detailed summary of the following text for a research purpose:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

    output = summary_chain.run(input_documents=docs,)

    return output