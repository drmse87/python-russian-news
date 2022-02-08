import csv
import os
import requests
from bs4 import BeautifulSoup

URL_INDEX = 0
CLASS_INDEX = 1
CSV_FILENAME = 'articles.csv'
DIR_NAME = 'extracted'

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

def extract_article(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    extracted_article_text_content = ''

    text_elements = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'p'])
    for element in text_elements:
        extracted_article_text_content = extracted_article_text_content + ' ' + element.get_text(separator=' ', strip=True)

    return extracted_article_text_content

def do_extract(articles):
    if not os.path.exists(DIR_NAME):
        os.makedirs(DIR_NAME)

    for index, article in enumerate(articles, start=1):
        article_class = article[CLASS_INDEX]
        article_url = article[URL_INDEX]

        output_filename = f"./{DIR_NAME}/{article_class}_{str(index)}.txt"
        output_content = extract_article(article_url)

        write_to_txt_file(output_filename, output_content)

def check_extracted_files():
    all_files = os.listdir(DIR_NAME)
    bad_files = []

    for file_name in all_files:
        with open(f'{DIR_NAME}/{file_name}', 'r', encoding='utf-8') as file_content:
            file_content_string = file_content.read()

            if len(file_content_string) < 300 or \
                'cookie' in file_content_string.lower() or \
                    'gdpr' in file_content_string.lower():
                bad_files.append(file_name)

    number_of_bad_files = len(bad_files)
    if number_of_bad_files > 0:
        print(f'{number_of_bad_files} files may be bad (length < 300 chars) and may need to be copied manually: ', end='')
        print(*bad_files, sep=', ')

if __name__ == '__main__':
    articles = read_csv_file()
    do_extract(articles)
    check_extracted_files()