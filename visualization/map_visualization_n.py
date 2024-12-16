import pandas as pd
import geopandas as gpd
import folium

n_df = pd.read_csv("../preprocessing_all/preprocessing_fianl_data_n.csv")  # 남양주 데이터 불러오기
korea_boundary = gpd.read_file("Z_SOP_BND_ADM_DONG_PG/Z_SOP_BND_ADM_DONG_PG.shp", encoding='cp949')  # 지도 데이터 불러오기

nam_boundary = korea_boundary[(korea_boundary['ADM_NM'] == '와부읍') | (korea_boundary['ADM_NM'] == '진접읍') | (korea_boundary['ADM_NM'] == '화도읍') |
                              (korea_boundary['ADM_NM'] == '진건읍') | (korea_boundary['ADM_NM'] == '오남읍') | (korea_boundary['ADM_NM'] == '퇴계원읍') |
                              (korea_boundary['ADM_NM'] == '별내면') | (korea_boundary['ADM_CD'] == '31130340') | (korea_boundary['ADM_NM'] == '조안면') |
                              (korea_boundary['ADM_NM'] == '호평동') | (korea_boundary['ADM_NM'] == '평내동') | (korea_boundary['ADM_CD'] == '31130530') |
                              (korea_boundary['ADM_CD'] == '31130540') | (korea_boundary['ADM_NM'] == '다산1동') | (korea_boundary['ADM_NM'] == '다산2동') |
                              (korea_boundary['ADM_NM'] == '별내동')]

boundary = nam_boundary[['ADM_NM', 'geometry']].set_index('ADM_NM')

m1 = folium.Map(location=[37.6, 127.15], tiles='cartodbpositron', zoom_start=11)

folium.Choropleth(
    geo_data=boundary['geometry'].geometry.to_crs(epsg=4326).__geo_interface__,
    data=n_df['병원'],
    key_on='feature.id',
    fill_color='YlGnBu',
    legend_name='hospital',
).add_to(m1)

m1.save('visualization_folium_hospital.html')
