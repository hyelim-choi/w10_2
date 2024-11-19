import pandas as pd
import time
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
url = 'https://map.naver.com/v5/search'
driver = webdriver.Chrome()
driver.get(url)
key_word = '남양주시 평내동 동물'
def time_wait(num, code):
    try:
        wait = WebDriverWait(driver, num).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, code)))
    except:
        print(code, '태그를 찾지 못하였습니다.')
        driver.quit()
    return wait
def switch_frame(frame):
    driver.switch_to.default_content()
    driver.switch_to.frame(frame)

def page_down(num):
    body = driver.find_element(By.CSS_SELECTOR, 'body')
    body.click()
    for i in range(num):
        body.send_keys(Keys.PAGE_DOWN)
time_wait(10, 'div.input_box > input.input_search')
search = driver.find_element(By.CSS_SELECTOR, 'div.input_box > input.input_search')
search.send_keys(key_word)  # 검색어 입력
search.send_keys(Keys.ENTER)  # 엔터버튼 누르기

sleep(1)

# (2) frame 변경
switch_frame('searchIframe')
page_down(40)
sleep(3)

dog_related_facilities_list = driver.find_elements(By.CSS_SELECTOR,'li.VLTHu')
next_btn = driver.find_elements(By.CSS_SELECTOR,'.zRM9F > a')
dog_facilities = []
start = time.time()
print('[크롤링 시작...]')

for btn in range(len(next_btn))[1:]:
    dog_related_facilities_list = driver.find_elements(By.CSS_SELECTOR, 'li.VLTHu')
    names = driver.find_elements(By.CSS_SELECTOR,'.YwYLL')
    types = driver.find_elements(By.CSS_SELECTOR,'.YzBgS')
    address = driver.find_elements(By.CSS_SELECTOR,'.lWwyx .Pb4bU')
    for data in range(len(dog_related_facilities_list)):
        print(data)
        sleep(1)
        try:
            dog_facilities_name = names[data].text
            print(dog_facilities_name)
            dog_facilities_type = types[data].text
            print(dog_facilities_type)
            address_name = address[data].text
            print(address_name)
            dog_facilities.append([dog_facilities_name, dog_facilities_type,address_name])
            print(f'{dog_facilities_name}...완료')
            sleep(1)
        except Exception as e:
            print(e)
            print('ERROR!'* 3)
            dog_facilities.append([dog_facilities_name, dog_facilities_type,address_name])
            print(f'{dog_facilities_name}...완료')
            sleep(1)
    if not next_btn[-1].is_enabled():
        break
    if names[-1]:
        next_btn[-1].click()
        sleep(2)
    else:
        print('페이지 인식 못함')
        break
print('[데이터 수집 완료]\n소요시간 :',time.time() - start)
driver.quit()
df = pd.DataFrame(dog_facilities,columns=['dog_facilities_name','dog_facilities_type','address_name'])
df.to_csv('개관련시설크롤링.csv',encoding='utf-8-sig')

