import csv
import os
import requests
from bs4 import BeautifulSoup

URL_INDEX = 0
CLASS_INDEX = 1
CSV_FILENAME = 'articles.csv'
DIR_NAME = 'scraped'

def read_csv_file():
    content = []

    with open(CSV_FILENAME, 'r') as file:
        csv_reader = csv.reader(file, delimiter=';')

        for row in csv_reader:
            content.append(row)
   
    return content

def write_to_txt_file(filename, content):
    txt_file = open(filename, 'w', encoding='utf-8')

    txt_file.write(content)
    txt_file.close() 

def scrape_article(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    extracted_article_text_content = ''

    text_elements = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'p'])
    for element in text_elements:
        extracted_article_text_content = extracted_article_text_content + ' ' + element.get_text(separator=' ', strip=True)

    return extracted_article_text_content

def do_scraping(articles):
    if not os.path.exists(DIR_NAME):
        os.makedirs(DIR_NAME)

    for index, article in enumerate(articles, start=1):
        article_class = article[CLASS_INDEX]
        article_url = article[URL_INDEX]

        output_filename = f"./{DIR_NAME}/{article_class}_{str(index)}.txt"
        output_content = scrape_article(article_url)

        write_to_txt_file(output_filename, output_content)

articles = read_csv_file()
do_scraping(articles)