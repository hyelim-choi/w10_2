import pandas as pd
import geopandas as gpd
import folium
import zipfile

zipfile.ZipFile('Z_SOP_BND_ADM_DONG_PG/Z_SOP_BND_ADM_DONG_PG.zip').extractall(path='Z_SOP_BND_ADM_DONG_PG')  # 지도 파일 압축해제

n_df = pd.read_csv("../preprocessing_all/preprocessing_final_data_n.csv")  # 남양주 데이터 불러오기
korea_boundary = gpd.read_file("Z_SOP_BND_ADM_DONG_PG/Z_SOP_BND_ADM_DONG_PG.shp", encoding='cp949')  # 지도 데이터 불러오기

n_df["all_facilities"] = n_df["미용"]+n_df["병원"]+n_df["약국"]+n_df["용품"]+n_df["위탁"]  # 전체 시설 열 추가

# 전체 지도에서 남양주만 추출
nam_boundary = korea_boundary[(korea_boundary['ADM_NM'] == '와부읍') | (korea_boundary['ADM_NM'] == '진접읍') | (korea_boundary['ADM_NM'] == '화도읍') |
                              (korea_boundary['ADM_NM'] == '진건읍') | (korea_boundary['ADM_NM'] == '오남읍') | (korea_boundary['ADM_NM'] == '퇴계원읍') |
                              (korea_boundary['ADM_NM'] == '별내면') | (korea_boundary['ADM_CD'] == '31130340') | (korea_boundary['ADM_NM'] == '조안면') |
                              (korea_boundary['ADM_NM'] == '호평동') | (korea_boundary['ADM_NM'] == '평내동') | (korea_boundary['ADM_CD'] == '31130530') |
                              (korea_boundary['ADM_CD'] == '31130540') | (korea_boundary['ADM_NM'] == '다산1동') | (korea_boundary['ADM_NM'] == '다산2동') |
                              (korea_boundary['ADM_NM'] == '별내동')]
boundary = nam_boundary[['ADM_NM', 'geometry']].set_index('ADM_NM')

# 지도 이름과 맞추기 위해 다산동을 다산1동, 다산2동으로 변경
df2 = n_df.loc[n_df['동'] == '남양주 다산동']
df2.loc[1, ['동']] = '다산1동'
df3 = n_df.loc[n_df['동'] == '남양주 다산동']
df3.loc[1, ['동']] = '다산2동'

n_df = pd.concat([n_df, df2, df3])  # 다산1동, 다산2동을 기존 데이터프레임과 합치기

n_df.index = ['금곡동', '다산동', '별내동', '별내면', '수동면', '오남읍', '와부읍', '조안면', '진건읍', '진접읍',
              '퇴계원읍', '평내동', '호평동', '화도읍', '양정동', '다산1동', '다산2동']  # 지도 이름과 같은 인덱스 설정

## 시설 지도 시각화
# 지도 중심 지역 설정
m1 = folium.Map(location=[37.6, 127.15], tiles='cartodbpositron', zoom_start=11)

# 단계구분도 만들기
folium.Choropleth(
    geo_data=boundary['geometry'].geometry.to_crs(epsg=4326).__geo_interface__,
    data=n_df['all_facilities'],  # 사용할 데이터 (시설)
    key_on='feature.id',  # 지도와 데이터를 id로 연결
    fill_color='YlGnBu',  # 색상
    legend_name='facilities',  # 범례
).add_to(m1)

m1.save('visualization_folium_facilities_n.html')  # 지도를 html로 저장

## 견주 수 지도 시각화
# 지도 중심 지역 설정
m2 = folium.Map(location=[37.6, 127.15], tiles='cartodbpositron', zoom_start=11)

# 단계구분도 만들기
folium.Choropleth(
    geo_data=boundary['geometry'].geometry.to_crs(epsg=4326).__geo_interface__,
    data=n_df['견주수'],  # 사용할 데이터 (견주수)
    key_on='feature.id',  # 지도와 데이터를 id로 연결
    fill_color='YlGnBu',  # 색상
    legend_name='facilities',  # 범례
).add_to(m2)

m2.save('visualization_folium_dogowner_n.html')  # 지도를 html로 저장
