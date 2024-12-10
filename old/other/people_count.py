import pandas as pd

df1 = pd.read_csv("경기도 남양주시_반려동물 등록현황(개)_20230205.csv", encoding="cp949")
df1["ratio"] = df1["등록소유자수"] / df1["세대수"] * 100

df1.to_csv("people_number.csv", encoding='utf-8-sig', index=False)
print(df1)
