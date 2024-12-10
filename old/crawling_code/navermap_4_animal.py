from selenium import webdriver
from bs4 import BeautifulSoup
import time
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys


def scroll_and_wait():
    body = browser.find_element_by_css_selector("body")
    for i in range(10):
        body.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.5)

        try:
            element_present = EC.presence_of_element_located((By.CSS_SELECTOR, ".VLTHu.OW9LQ"))
            WebDriverWait(browser, 10).until(element_present)
        except TimeoutException:
            pass


browser = webdriver.Chrome("chromedriver.exe")

# 남양주 조안면 동물
browser.get("https://map.naver.com/p/search/%EB%82%A8%EC%96%91%EC%A3%BC%20%EC%A1%B0%EC%95%88%EB%A9%B4%20%EB%8F%99%EB%AC%BC?c=12.00,0,0,0,dh")  # # #
time.sleep(3)
html = browser.page_source
soup = BeautifulSoup(html, "html.parser")

map_area = soup.select_one("#searchIframe").get('src')
print(map_area)

browser.get(map_area)  # 크롤링 가능한 주소
html = browser.page_source
soup = BeautifulSoup(html, "html.parser")
time.sleep(3)

result = []
for i in range(6):
    if (i == int(soup.select(".mBN2s")[-1].text)) or (i == 6):
        break
    print(i+1, "페이지")

    try:
        browser.find_element(By.CSS_SELECTOR, ".VLTHu.OW9LQ").click()
    except Exception as e:
        print("예외:", e)
        break
    scroll_and_wait()  # 스크롤
    html = browser.page_source
    soup = BeautifulSoup(html, "html.parser")

    map_list = soup.select(".Ryr1F .VLTHu.OW9LQ")
    print(len(map_list))

    for map_one in map_list:
        map_data = map_one.select_one(".ouxiq")
        name = map_data.select_one(".YwYLL").text
        kate = map_data.select_one(".YzBgS").text
        address = map_data.select_one(".Pb4bU").text

        if address == "남양주 조안면":  # # #
            result.append([name, kate, address])
            print(result[-1])

    browser.find_elements(By.CSS_SELECTOR, ".eUTV2")[1].click()  # 다음페이지
    html = browser.page_source
    soup = BeautifulSoup(html, "html.parser")
    time.sleep(3)

# 지도 정보
df = pd.DataFrame(result, columns=["dog_facilities_name", "dog_facilities_type", "address_name"])
df.to_csv("남양주 조안면 동물.csv", encoding='utf-8-sig')  # # #
print(df)
