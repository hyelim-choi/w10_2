import pandas as pd

result = []

df1 = pd.read_csv("남양주 퇴계원읍 동물_dog.csv")
print(df1["re_dog_facilities_type"].value_counts())
a = len(df1.loc[df1["re_dog_facilities_type"] == "병원"])
b = len(df1.loc[df1["re_dog_facilities_type"] == "용품"])
c = len(df1.loc[df1["re_dog_facilities_type"] == "장례"])
d = len(df1.loc[df1["re_dog_facilities_type"] == "약국"])
e = len(df1.loc[df1["re_dog_facilities_type"] == "위탁"])
f = len(df1.loc[df1["re_dog_facilities_type"] == "분양"])
g = len(df1.loc[df1["re_dog_facilities_type"] == "복합시설"])
h = len(df1.loc[df1["re_dog_facilities_type"] == "놀이터"])
i = len(df1.loc[df1["re_dog_facilities_type"] == "사진"])
j = len(df1.loc[df1["re_dog_facilities_type"] == "미용"])
result.append(["퇴계원읍", a, b, c, d, e, f, g, h, i, j])

df2 = pd.read_csv("남양주 별내면 동물_dog.csv")
print(df2["re_dog_facilities_type"].value_counts())
a = len(df2.loc[df2["re_dog_facilities_type"] == "병원"])
b = len(df2.loc[df2["re_dog_facilities_type"] == "용품"])
c = len(df2.loc[df2["re_dog_facilities_type"] == "장례"])
d = len(df2.loc[df2["re_dog_facilities_type"] == "약국"])
e = len(df2.loc[df2["re_dog_facilities_type"] == "위탁"])
f = len(df2.loc[df2["re_dog_facilities_type"] == "분양"])
g = len(df2.loc[df2["re_dog_facilities_type"] == "복합시설"])
h = len(df2.loc[df2["re_dog_facilities_type"] == "놀이터"])
i = len(df2.loc[df2["re_dog_facilities_type"] == "사진"])
j = len(df2.loc[df2["re_dog_facilities_type"] == "미용"])
result.append(["별내면", a, b, c, d, e, f, g, h, i, j])

df1 = pd.read_csv("남양주 수동면 동물_dog.csv")
print(df1["re_dog_facilities_type"].value_counts())
a = len(df1.loc[df1["re_dog_facilities_type"] == "병원"])
b = len(df1.loc[df1["re_dog_facilities_type"] == "용품"])
c = len(df1.loc[df1["re_dog_facilities_type"] == "장례"])
d = len(df1.loc[df1["re_dog_facilities_type"] == "약국"])
e = len(df1.loc[df1["re_dog_facilities_type"] == "위탁"])
f = len(df1.loc[df1["re_dog_facilities_type"] == "분양"])
g = len(df1.loc[df1["re_dog_facilities_type"] == "복합시설"])
h = len(df1.loc[df1["re_dog_facilities_type"] == "놀이터"])
i = len(df1.loc[df1["re_dog_facilities_type"] == "사진"])
j = len(df1.loc[df1["re_dog_facilities_type"] == "미용"])
result.append(["수동면", a, b, c, d, e, f, g, h, i, j])

df4 = pd.read_csv("남양주 조안면 동물_dog.csv")
print(df4["re_dog_facilities_type"].value_counts())
a = len(df4.loc[df4["re_dog_facilities_type"] == "병원"])
b = len(df4.loc[df4["re_dog_facilities_type"] == "용품"])
c = len(df4.loc[df4["re_dog_facilities_type"] == "장례"])
d = len(df4.loc[df4["re_dog_facilities_type"] == "약국"])
e = len(df4.loc[df4["re_dog_facilities_type"] == "위탁"])
f = len(df4.loc[df4["re_dog_facilities_type"] == "분양"])
g = len(df4.loc[df4["re_dog_facilities_type"] == "복합시설"])
h = len(df4.loc[df4["re_dog_facilities_type"] == "놀이터"])
i = len(df4.loc[df4["re_dog_facilities_type"] == "사진"])
j = len(df4.loc[df4["re_dog_facilities_type"] == "미용"])
result.append(["조안면", a, b, c, d, e, f, g, h, i, j])

df5 = pd.read_csv("남양주 호평동 동물_dog.csv")
print(df5["re_dog_facilities_type"].value_counts())
a = len(df5.loc[df5["re_dog_facilities_type"] == "병원"])
b = len(df5.loc[df5["re_dog_facilities_type"] == "용품"])
c = len(df5.loc[df5["re_dog_facilities_type"] == "장례"])
d = len(df5.loc[df5["re_dog_facilities_type"] == "약국"])
e = len(df5.loc[df5["re_dog_facilities_type"] == "위탁"])
f = len(df5.loc[df5["re_dog_facilities_type"] == "분양"])
g = len(df5.loc[df5["re_dog_facilities_type"] == "복합시설"])
h = len(df5.loc[df5["re_dog_facilities_type"] == "놀이터"])
i = len(df5.loc[df5["re_dog_facilities_type"] == "사진"])
j = len(df5.loc[df5["re_dog_facilities_type"] == "미용"])
result.append(["호평동", a, b, c, d, e, f, g, h, i, j])

df = pd.DataFrame(result,
                  columns=["행정동", "hospital", "product", "funeral", "pharmacy", "foster_place", "adoption", "complex", "playground", "photo", "beauty"])
df.to_csv("type_number.csv", encoding='utf-8-sig', index=False)
print(df)
