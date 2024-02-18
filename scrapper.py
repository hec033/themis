"""
Created on: Wed Feb 14 2024
@author: Heeje Cho

Useful Links:
https://oxylabs.io/blog/python-web-scraping
https://www.nysenate.gov/legislation/laws/CONSOLIDATED
"""

import os
import time
import pandas as pd
import requests
import tiktoken
from bs4 import BeautifulSoup
from typing import List
from pinecone import Pinecone, ServerlessSpec, PodSpec
import hashlib
from openai import OpenAI
import json


# url = 'https://www.nysenate.gov/legislation/laws/ABC/10'
# response = requests.get(url)
# soup = BeautifulSoup(response.text, 'html.parser')
#
# article = soup.find_all("div", {"class": "nys-openleg-result-nav-item-name"})[0].text
# article_title = soup.find_all("div", {"class": "nys-openleg-result-nav-item-description"})[0].text
# section_number = soup.find_all("div", {"class": "nys-openleg-result-title-headline"})[0].text
# section_title = soup.find_all("div", {"class": "nys-openleg-result-title-short"})[0].text
# section_title_location = soup.find_all("div", {"class": "nys-openleg-result-title-location"})[0].text
# section_content = soup.find_all("div", {"class": "nys-openleg-result-text"})[0].text


def retrieve_all_nyc_law_urls(url: str):
    # should be https://www.nysenate.gov/legislation/laws
    response = requests.get(url)

    return None


def get_embedding(openai_client, text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")

    return openai_client.embeddings.create(input=[text], model=model).data[0].embedding


def extract_encode_law_content(openai_client, url: str):
    # define encoder ("cl100k_base")
    # encoding = tiktoken.encoding_for_model("text-embedding-ada-002")

    # using requests grab html content from url
    response = requests.get(url)

    # extract html doc using parser
    soup = BeautifulSoup(response.text, 'html.parser')

    # format dictionary of relevent content (section, title_short, title_location, content
    section_dict = {
        "metadata": {
            "state": soup.find(class_='nys-openleg-result-breadcrumbs-container').find_all(class_="nys-openleg-result-breadcrumb-container")[0].find(class_='nys-openleg-result-breadcrumb-name').text.strip(),
            "legislation": soup.find(class_='nys-openleg-result-breadcrumbs-container').find_all(class_="nys-openleg-result-breadcrumb-container")[1].find(class_='nys-openleg-result-breadcrumb-name').text.strip(),
            "chapter": soup.find(class_='nys-openleg-result-breadcrumbs-container').find_all(class_="nys-openleg-result-breadcrumb-container")[2].find(class_='nys-openleg-result-breadcrumb-name').text.strip(),
            "chapter_title": soup.find(class_='nys-openleg-result-breadcrumbs-container').find_all(class_="nys-openleg-result-breadcrumb-container")[2].find(class_='nys-openleg-result-breadcrumb-description').text.strip(),
            "article": soup.find(class_='nys-openleg-result-breadcrumbs-container').find_all(class_="nys-openleg-result-breadcrumb-container")[3].find(class_='nys-openleg-result-breadcrumb-name').text.strip(),
            "article_title": soup.find(class_='nys-openleg-result-breadcrumbs-container').find_all(class_="nys-openleg-result-breadcrumb-container")[3].find(class_='nys-openleg-result-breadcrumb-description').text.strip(),
            "section": soup.find_all("div", {"class": "nys-openleg-result-title-headline"})[0].text.strip(),
            "section_title": soup.find_all("div", {"class": "nys-openleg-result-title-short"})[0].text.strip(),
            "section_content": soup.find_all("div", {"class": "nys-openleg-result-text"})[0].get_text(" ").strip()
        },
        "values": get_embedding(openai_client, soup.find_all("div", {"class": "nys-openleg-result-text"})[0].get_text(" ").strip(), model="text-embedding-ada-002")
    }

    # replace content with breadcrumbs added
    section_dict["metadata"]["section_content"] = section_dict["metadata"]["state"] + '\n' + section_dict["metadata"]["legislation"] + '\n' + \
                                                  section_dict["metadata"]["chapter"] + ': ' + section_dict["metadata"]["chapter_title"] + '\n' + \
                                                  section_dict["metadata"]["article"] + ': ' + section_dict["metadata"]["article_title"] + '\n' + \
                                                  section_dict["metadata"]["section"] + ': ' + section_dict["metadata"]["section_title"] + '\n' + section_dict["metadata"]["section_content"]

    # create id by hashing MD5 metadata together
    metadata_hashable = section_dict["metadata"]["state"] + '-' + section_dict["metadata"]["legislation"] + '-' + \
                        section_dict["metadata"]["chapter"] + section_dict["metadata"]["article"] + section_dict["metadata"]["section"]

    section_dict["id"] = hashlib.md5(metadata_hashable.encode()).hexdigest()

    return section_dict


def convert_to_law_vectors(url_list: List[str]):
    # instantiate OpenAI Client
    client = OpenAI(api_key=os.environ["OPENAI_THEMIS_KEY"])

    # empty dictionary to add to
    dict_list = []

    # call extract_law_content function to extract relevant info from web and append to dictionary list
    for url in url_list:
        law_dict = extract_encode_law_content(client, url)
        dict_list.append(law_dict)

    return dict_list


def upsert_encoding_to_pinecone(index_name: str, vectors: List[dict]):
    # initialize connection to pinecone
    api_key = os.environ.get("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)

    # check if index already exists (themis-gpt-4-turbo-preview)
    if index_name not in pc.list_indexes().names():
        # if it does not create, create index
        pc.create_index(
            index_name,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-west-2')
        )
        # wait for index to be initialized
        time.sleep(1)
    # connect to index
    index = pc.Index(index_name)

    # view index stats
    # index.describe_index_stats()

    # upsert records to pinecone
    upsert_response = index.upsert(vectors=vectors)

    return upsert_response


def main():
    # TODO: need to recursively grab a list of all scrappable URLs from https://www.nysenate.gov/legislation/laws
    law_urls = [
        'https://www.nysenate.gov/legislation/laws/ABC/10',
        'https://www.nysenate.gov/legislation/laws/ABC/11',
        'https://www.nysenate.gov/legislation/laws/ABC/12',
        'https://www.nysenate.gov/legislation/laws/ABC/13',
        'https://www.nysenate.gov/legislation/laws/ABC/14',
        'https://www.nysenate.gov/legislation/laws/ABC/15',
        'https://www.nysenate.gov/legislation/laws/ABC/16',
        'https://www.nysenate.gov/legislation/laws/ABC/17',
        'https://www.nysenate.gov/legislation/laws/ABC/18',
        'https://www.nysenate.gov/legislation/laws/ABC/19',
    ]

    # extracting relevant information from urls into pandas dataframe
    law_vectors = convert_to_law_vectors(law_urls)

    # upsert the vectors to Pinecone database
    response = upsert_encoding_to_pinecone(index_name='themis-gpt-4-turbo-preview', vectors=law_vectors)

    print(response)

    # TODO: Check for when last updated and which ones to refresh

    # TODO: Load into Iceberg tables

    # TODO: Vectorize content (section_content) into tokens

    # TODO: DAG all this


if __name__ == '__main__':
    main()
