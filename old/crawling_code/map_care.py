from selenium import webdriver
from bs4 import BeautifulSoup
import time
import pandas as pd

browser = webdriver.Chrome("chromedriver.exe")

browser.get("https://www.silvercarekorea.com/silver/list.php?addcode=41360#google_vignette")  # # #
time.sleep(3)
html = browser.page_source
soup = BeautifulSoup(html, "html.parser")

n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15 = [0 for _ in range(15)]
while True:
    map_area = soup.select(".datatable12 > tbody > tr")
    map_area = map_area[1:]

    for map_one in map_area:
        map_data = map_one.select("td")[1].select("div")
        map_address = map_data[1].select_one("a").text
        if "와부읍" in map_address:
            n1 += 1
        elif "진접읍" in map_address:
            n2 += 1
        elif "화도읍" in map_address:
            n3 += 1
        elif "진건읍" in map_address:
            n4 += 1
        elif "오남읍" in map_address:
            n5 += 1
        elif "퇴계원읍" in map_address:
            n6 += 1
        elif "별내면" in map_address:
            n7 += 1
        elif "수동면" in map_address:
            n8 += 1
        elif "조안면" in map_address:
            n9 += 1
        elif "호평동" in map_address:
            n10 += 1
        elif "평내동" in map_address:
            n11 += 1
        elif "금곡동" in map_address:
            n12 += 1
        elif "양정동" in map_address:
            n13 += 1
        elif "다산동" in map_address:
            n14 += 1
        elif "별내동" in map_address:
            n15 += 1
        else:
            print(map_address)

    try:
        next_area = soup.select(".datatable20 > tbody > tr > td")[2].select_one("a").get("href")
        browser.get("https://www.silvercarekorea.com/silver/" + next_area)  # # #
        time.sleep(3)
        html = browser.page_source
        soup = BeautifulSoup(html, "html.parser")
    except Exception as e:
        print("예외:", e)
        break

result = [["와부읍", n1], ["진접읍", n2], ["화도읍", n3], ["진건읍", n4], ["오남읍", n5],
          ["퇴계원읍", n6], ["별내면", n7], ["수동면", n8], ["조안면", n9], ["호평동", n10],
          ["평내동", n11], ["금곡동", n12], ["양정동", n13], ["다산동", n14], ["별내동", n15]]
