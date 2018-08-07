'''
Use this script to get item release date from Google and Wikipedia
'''

import requests
import pandas as pd
import re

fpath = "./data-readonly/items.csv"
df_Items = pd.read_csv(fpath)

query_re = re.compile(r'[(\[].*?[)\]]')
wiki_re = re.compile(r'^.*wiki\/(.*)$')
date_re = re.compile(r'\|(\d{4})\|(\d{1,2})\|(\d{1,2})')
n = df_Items.shape[0]

for i in range(n):
    print('fetching {}/{}   {:.2%}'.format(i,n,i/n))
    query = df_Items.loc[i].item_name
    query = query_re.sub('',query)

    api_key = 'AIzaSyAQ22rvMB0RzfNcnlJ9yYEKQJPW9w_jF8k'
    proxy = {'http':'http://127.0.0.1:6152',
             'https': 'http://127.0.0.1:6152'}
    service_url = 'https://kgsearch.googleapis.com/v1/entities:search'
    params = {
        'query': query,
        'indent': True,
        'key': api_key,
        'languages':['en', 'ru'],
        'limit': 1,
    }
    s = requests.Session()
    s.proxies = proxy


    try:
        response = s.get(service_url, params=params, timeout=100).json()
        wiki_url = response['itemListElement'][0]['result']['detailedDescription'][0]['url']
        wiki_key = wiki_re.search(wiki_url).group(1)
        params = {
            'action': 'query',
            'titles': wiki_key,
            'prop': 'revisions',
            'rvprop': 'content',
            'format': 'jsonfm',
            'rvsection': 0,

        }
        headers = {
            'User-Agent': '1C_Company_competition 1.0',
            'From': 'qfuxiang@gmail.com'
        }
        wiki_api = 'https://en.wikipedia.org/w/api.php'
        response = s.get(wiki_api, params=params, headers=headers, timeout=100).text
        date = date_re.search(response).group(1) + '-' + date_re.search(response).group(2) + '-' + date_re.search(response).group(3)
    except:
        date = None
    df_Items.loc[i, 'release_date'] = date

df_Items.to_csv('items_with_release_date.csv')