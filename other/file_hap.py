import pandas as pd

df1 = pd.read_csv("통합 문서1.csv")
df2 = pd.read_csv("type_number.csv")
df1 = pd.read_csv("merged_result.csv")

df_all = pd.concat([df1, df2, df1])

df_all.to_csv("all_type_number.csv", encoding='utf-8-sig', index=False)
print(df_all)

df1 = pd.read_csv("all_type_number.csv")
df2 = pd.read_csv("people_number_area.csv")

df1 = pd.concat([df1, df2[df2.columns[-2:]]], axis=1)
df1["all_facilities"] = df1[["hospital", "product", "pharmacy", "foster_place", "beauty"]].sum(axis=1)

df1.to_csv("all.csv", encoding='utf-8-sig', index=False)

print(df1)
