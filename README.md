# 반려동물 관련 기관과 반려동물 수의 관계 예측
파이썬을 이용한 데이터 분석 연계실습2 2조

## 👨‍🏫 1. 프로젝트 설명
  캡스톤1의 주제였던 "펫티켓 수요 예측"을 하면서 반려동물 관련 기관의 수와 반려동물의 수가 연관이 있다고 생각, 반려동물 기관의 수를 이용하여 지역별 견주 수를 예측하는 모델을 학습함
  
## ⚙️ 2. 프로젝트 구성 (레파지토리 설명)
  - crawling_code_changwon
    
    : 창원시의 읍면동 별 반려동물 관련 기관 및 견주 수 확인을 위한 자료
    - changwon_data.csv: 창원시 반려동물 등록소유자수 공공데이터
    - crawling_c.py: 창원시 크롤링 코드
  - crawling_code_nan
    
    : 남양주시의 읍면동 별 반려동물 관련 기관 및 견주 수 확인을 위한 자료
    - crawling_n.py: 남양주시 크롤링 코드
    - nan_data.csv: 남양주시 반려동물 등록소유자수 공공데이터
  - crawling_file_changwon
  
    : 창원시의 반려동물 관련 기관 크롤링한 데이터 저장
    - changwon_facilities_all.csv: 창원시 크롤링 결과
  - crawling_file_nan
  
    : 남양주의 반려동물 관련 기관 크롤링한 데이터 저장
    - changwon_facilities_all.csv: 남양주시 크롤링 결과
  - modeling
  
    : 창원시의 데이터를 학습시켜 남양주의 반려동물 수 예측 7가지 방법
    - modeling.py: 모델링 코드
  - old
  
    : 캡스톤1에서 사용한 코드들 정리
  - preprocessing_all
  
    : 데이터 전처리한 부분 최종
    - preprocessing_final_data_c.csv: 시각화 및 모델 학습을 위한 최종 데이터
    - preprocessing_final_data_n.csv: 시각화 및 모델 평가를 위한 최종 데이터
  - visualization
  
    : 창원시와 남양주의 반려동물 시설 별 반려인 수, 지도 시각화 최종
    - Z_SOP_BND_ADM_DONG_PG: 시각화를 위한 지도 데이터
    - graph_visualization_c.py: 창원시의 반려동물 시설 별 반려인 수 시각화
    - graph_visualization_n.py: 남양주시의 반려동물 시설 별 반려인 수 시각화
    - map_visualization_n.py: 타겟 데이터(남양주시)의 시설, 견주 수에 따른 지도 시각화
    
## 💻 3. 프로젝트 프로그램 사용법
    - main/modeling/modeling.py 실행했을 때 데이터 시각화 및 각 모델의 예측 결과 확인 가능
    
    - main/visualization/graph_visualization_c.py 실행했을 때 창원시의 지역별 반려동물 시설 수에 따른 견주 수 그래프 확인 가능 
    - main/visualization/graph_visualization_n.py 실행했을 때 남양주시의 지역별 반려동물 시설 수에 따른 견주 수 그래프 확인 가능 
    - main/visualization/map_visualization_n.py 실행했을 때 생성되는 html -> 파일 경로를 찾아서 직접 파일 실행시 남양주시의 지도 시각화 확인 가능
## 📌 4. modeling/modeling.py 코드 5가지 설명
  - version 01
    : Linear Regression 모델 학습
  - version 1.2
    : Linear Regression 모델에 SVM, DT 모델 추가


    
    릿지 랏소 디시전트리 SGD 모델 예측 결과 시각화/히트맵

## 📈 5. 캡스톤에서 발전된 부분
  ### - 데이터 크롤링
      - before
        : 공공데이터의 기재된 읍면동을 각자 나눠서 크롤링
          검색어를 "남양주시 읍면동별 동물"로 설정했기에 크롤링 할 때마다 자체적으로 수정
      - after
        : 추가로 창원시에 관한 시설을 크롤링
          공공데이터에서 제공하는 CSV파일을 이용하여 검색어를 수정하지 않아도 크롤링 가능
  ### - 데이터 전처리
      - before
        : 크롤링된 데이터를 직접 분류, 중복 확인 및 타입 분류, 코드 사용을 안함
      - after
        : 코드를 이용한 전처리
          관련 없는 가게 제거, 주소가 창원 또는 남양주시가 아닌 데이터 제거
          유사도를 이용한 타입 분류
          다만, 원하는 5가지로 분류되지 않기에 추가로 직접 분류
  ### - 데이터 시각화
      - before
        : 남양주시의 데이터만 시각화
      - after
        : 남양주시의 데이터에 새로 수집한 창원시의 데이터 시각화 추가
        시각화 결과를 명확하게 확인하기 위해 견주 수의 스케일 재조정

  ### - 모델링
      - before
        : 
      - after
        : 

## 👩‍👩‍👧‍👧 6. 조원 정보
  학과 | 이름
  --- | ---
  👩‍💻 디지털미디어학과 | 김연주
  👩‍💻 데이터사이언스학과 | 이승윤
  👩‍💻 데이터사이언스학과 | 이윤경
  👩‍💻 디지털미디어학과 | 최혜림





    
