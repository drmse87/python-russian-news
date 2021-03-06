import csv
import os
import re
import requests
from bs4 import BeautifulSoup

ARTICLE_ID_INDEX = 0
CLASS_INDEX = 3
URL_INDEX = 4
CSV_FILENAME = 'articles.csv'
DIR_NAME = 'extracted'

unwanted_classes_rt = ['article__share', 'article__share_bottom', 'article__short-url', 'article__google-news', 
    'article__tags-trends', 'langs__item', 'article__date', 'read-more', 'read-more-big__container', 
    'news-block', 'header__section', 'article__breadcrumbs', 'header-where-schedule', 
    'header__rtshop', 'nav__item', 'cookies__banner-wrapper', 'main-page-podcasts-white',
    'tags-public__link', 'subscribe', 'footer', 'rtcode']
article_text_content_classes_rt = ['article__heading', 'article__summary', 'article__text']
unwanted_classes_sputnik = ['b-article__refs-credits', 'b-article__refs-text', 
    'article__google-news', 'comments__tabs-content', 'social-likes-pane', 'comments section', 
    'b-article__refs-item', 'b-article__likes', 'b-article__meta', 'ria-tweet']
article_text_content_classes_sputnik = ['b-article__header-title', 'b-article']

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

def extract_article(url, unwanted_classes, article_text_content_classes):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')

    decompose_unwanted_classes(soup, unwanted_classes)
    decompose_unwanted_links(soup)

    return extract_article_text_content(soup, article_text_content_classes)

def extract_article_text_content(soup, article_text_content_classes):
    article_text_content_classes = soup.find_all(class_=article_text_content_classes)
    extracted_article = ''

    for t in article_text_content_classes:
        extracted_article = extracted_article + " " + t.get_text(separator=' ', strip=True)

    return extracted_article

def decompose_unwanted_classes(soup, unwanted_classes):
    unwanted_elements_by_class = soup.find_all(class_=unwanted_classes)

    for c in unwanted_elements_by_class:
        c.decompose()

def decompose_unwanted_links(soup):
    read_more_links = soup.find_all('strong', string=re.compile('READ MORE.*'))
    like_this_story_links = soup.find_all('em', string=re.compile('Like this story.*'))
    urls = soup.find_all('a', string=re.compile('https:\/\/.*'))

    for e in read_more_links + like_this_story_links + urls:
        e.parent.decompose()

def do_extract(articles):
    if not os.path.exists(DIR_NAME):
        os.makedirs(DIR_NAME)

    for article in articles:
        article_id = article[ARTICLE_ID_INDEX]
        article_class = article[CLASS_INDEX]
        article_url = article[URL_INDEX]
        output_filename = f"./{DIR_NAME}/{article_class}_{article_id}.txt"
        output_content = ''

        if 'sputnik' in article_id:
            output_content = extract_article(article_url, unwanted_classes_sputnik, article_text_content_classes_sputnik)
        elif 'rt' in article_id:
            output_content = extract_article(article_url, unwanted_classes_rt, article_text_content_classes_rt)

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