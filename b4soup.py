import requests
from bs4 import BeautifulSoup
import csv
# url="https://sp4n.lapor.go.id/kisah-sukses?page=1"
# r=requests.get(url)
# soup = BeautifulSoup(r.content)
#
# s = soup.find_all('div', class_='post-excerpt')
# b = soup.find_all('div', class_='author-name')
filename = 'scraped_full.csv'
import pickle


datas=[]
for j in range(1,76) :
    url = f"https://sp4n.lapor.go.id/kisah-sukses?page={j}"
    r = requests.get(url)
    soup = BeautifulSoup(r.content)
    if r :
        s = soup.find_all('div', class_='post-excerpt')
        b = soup.find_all('div', class_='author-name')
        pt=soup.find_all('div', class_='post-title h4 mg-0')
        for i in range(len(s)) :
            text=s[i].find('p').get_text().strip()
            text2=b[i].find('a').get_text()
            text3=pt[i].find('a').get_text().lstrip().strip()
            data = {}
            data['author_name']=text2
            data['judul_keluhan'] = text3
            data['detail_keluhan']=text
            datas.append(data)
        print(f"page {j} printed")
# Open a file and use dump()
with open('saved_data.pkl', 'wb') as file:
    # A new file will be created
    pickle.dump(datas, file)
with open(filename, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['author_name', 'judul_keluhan' , 'detail_keluhan']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for dat in datas:
        writer.writerow(dat)