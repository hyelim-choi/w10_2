# w10_2
파이썬을 이용한 데이터 분석 연계실습2 2조

## 1. 프로젝트 설명
  캡스톤1의 주제였던 "펫티켓 수요 예측"에서 더 나아가 __반려동물 관련 기관과 반려동물의 수의 관계 예측__
  
## 2. 프로젝트 구성 (브랜치 설명)
  ### - main:
      - crawling_code_changwon: 창원시의 읍면동 별 반려동물 관련 기관 크롤링
      - crawling_code_nan: 남양주의 읍면동 별 반려동물 관련 기관 크롤링
      - crawling_file_changwon: 창원시의 반려동물 관련 기관 크롤링한 데이터 저장
      - crawling_file_nan: 남양주의 반려동물 관련 기관 크롤링한 데이터 저장
      - modeling: 창원시의 데이터를 학습시켜 남양주의 반려동물 수 예측 5가지 방법
      - old: 캡스톤1에서 사용한 코드들 정리
      - preprocessing_all: 데이터 전처리한 부분 최종
      - visualization: 창원시와 남양주의 반려동물 시설 별 반려인 수, 지도 시각화 최종
        
  ### - crawling ChangWon:
    창원시의 읍면동 별 반려동물 관련 기관 크롤링 코드
  ### - crawling:
    남양주의 읍면동 별 반려동물 관련 기관 크롤링 코드
  ### - preprocessing:
    데이터 전처리(남양주의 반려동물 수, 관련 기관 분리, 창원시의 반려동물 수, 관련 기관 분리)
  ### - visualization:
    창원시와 남양주의 반려동물 시설 별 반려인 수, 지도 시각화
    
## 3. 프로젝트 프로그램 사용법
    - main/modeling/new.py 실행했을 때 각 모델의 예측 결과 시각화 및 히트맵 확인 가능
    
    - main/visualization/map_visualization_n.py 실행했을 때 생성되는 html -> 파일 경로를 찾아서 직접 파일 실행시 지도 시각화 확인 가능
## 4. main/modeling 코드 5개 설명
  - version 01
    : Linear Regression 모델 학습
  - version 1.2
    : Linear Regression 모델에 SVM, DT 모델 추가

    
    릿지 랏소 디시전트리 SGD 모델 예측 결과 시각화/히트맵

## 5. 캡스톤에서 발전된 부분
  - 

## 6. 조원 정보
    👩‍💻 디지털미디어학과 김연주
    👩‍💻 데이터사이언스학과 이승윤
    👩‍💻 데이터사이언스학과 이윤경
    👩‍💻 디지털미디어학과 최혜림
