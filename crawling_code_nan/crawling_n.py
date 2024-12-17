from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from time import sleep

# CSV 파일
df = pd.read_csv('nan_data.csv', encoding='utf-8')  # CSV 파일 경로로 수정

unique_cities = df['읍면동명'] # 추후 검색어 설정 시 필요

# 검색어 기반 URL
url = 'https://map.naver.com/v5/search'


# 함수를 통해 새 브라우저를 실행
def start_browser():
    driver = webdriver.Chrome()
    driver.get(url)
    return driver


# 요소 대기 함수
def time_wait(driver, num, code):
    try:
        wait = WebDriverWait(driver, num).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, code))
        )
        return wait
    except:
        print(f"'{code}' 태그를 찾지 못하였습니다.")
        return None


# Frame 전환
def switch_frame(driver, frame):
    driver.switch_to.default_content()
    driver.switch_to.frame(frame)


# 페이지 스크롤
def page_down(driver, num):
    body = driver.find_element(By.CSS_SELECTOR, 'body')
    body.click()
    for i in range(num):
        body.send_keys(Keys.PAGE_DOWN)


# 결과 저장 경로
output_folder = '../crawling_file_nan'
dog_facilities_all = []

# 읍면동별 검색 및 크롤링
for city in unique_cities:
    key_word = f"남양주시 {city} 동물"
    print(f"\n[검색어]: {key_word}")

    # 새로운 브라우저 시작
    driver = start_browser()

    # 검색어 입력
    wait_result = time_wait(driver, 10, 'div.input_box > input.input_search')
    if not wait_result:
        print(f"검색어 입력 실패: {key_word}")
        driver.quit()
        continue

    search = driver.find_element(By.CSS_SELECTOR, 'div.input_box > input.input_search')
    search.clear()
    search.send_keys(key_word)
    search.send_keys(Keys.ENTER)
    sleep(1)

    # Frame 전환
    switch_frame(driver, 'searchIframe')
    page_down(driver, 40)
    sleep(3)

    # 크롤링 데이터 저장
    dog_related_facilities_list = driver.find_elements(By.CSS_SELECTOR, 'li.VLTHu')
    names = driver.find_elements(By.CSS_SELECTOR, '.YwYLL') # 시설 이름
    types = driver.find_elements(By.CSS_SELECTOR, '.YzBgS') # 시설 종류
    addresses = driver.find_elements(By.CSS_SELECTOR, '.lWwyx .Pb4bU') # 시설 주소

    for i in range(len(dog_related_facilities_list)):
        try:
            dog_facilities_name = names[i].text
            dog_facilities_type = types[i].text
            address_name = addresses[i].text
            # dog_facilities_all에 데이터 추가
            dog_facilities_all.append([dog_facilities_name, dog_facilities_type, address_name])
        except Exception as e:
            print(f"데이터 추출 오류: {e}")
            continue

    # 브라우저 종료
    driver.quit()

# 모든 데이터를 하나의 DataFrame으로 변환
df = pd.DataFrame(dog_facilities_all, columns=['dog_facilities_name', 'dog_facilities_type', 'address_name'])

# 하나의 파일로 저장
output_file = f"{output_folder}/nan_facilities_all.csv"
df.to_csv(output_file, encoding='utf-8-sig', index=False)
print(f"[저장 완료]: {output_file}")

