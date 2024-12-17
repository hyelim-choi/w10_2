from collections import defaultdict, Counter
import pandas as pd
from difflib import SequenceMatcher

# CSV 파일 로드
nan_df = pd.read_csv('../crawling_file_nan/nan_facilities_all.csv')

# 이상한 가게 제거 : 주소가 다른데 이름은 같은 데이터 제거
changwon_df = changwon_df[~changwon_df.duplicated(subset=['dog_facilities_name'], keep=False) | (changwon_df.duplicated(subset=['address_name'], keep=False))]

# 창원 주소 필터링
changwon_df = changwon_df[changwon_df['address_name'].str.contains('창원', na=False)]

# 중복 제거
changwon_df = changwon_df.drop_duplicates()

# '구' 단위로 묶기 위한 처리
def extract_district(address):
    # 주소에서 구를 추출 (예: '창원 성산구' -> '성산구')
    parts = address.split()
    for part in parts:
        if '구' in part:
            return part
    return None  # 구를 찾지 못한 경우

changwon_df['district'] = changwon_df['address_name'].apply(extract_district)

# 구별로 유사한 dog_facilities_type 그룹화
def find_similar_groups(types):
    """
    문자열 유사도를 기준으로 비슷한 값끼리 그룹화합니다.
    """
    types = types.copy()  # 원본 리스트를 복사
    groups = []
    while types:
        base = types.pop(0)
        group = [base]
        remaining_types = []
        for t in types:
            if SequenceMatcher(None, base, t).ratio() >= 0.7:  # 유사도 기준
                group.append(t)
            else:
                remaining_types.append(t)
        groups.append(group)
        types = remaining_types  # 업데이트
    return groups

# 새로운 타입 매핑 저장
new_type_mapping = {}

# 그룹화된 데이터 생성 (구별로)
grouped_data = {}
for district, group in changwon_df.groupby('district'):
    facilities = group['dog_facilities_type'].dropna().tolist()
    similar_groups = find_similar_groups(facilities)
    grouped_data[district] = similar_groups
    # 새로운 타입 매핑
    for group in similar_groups:
        # 그룹에서 가장 많이 등장한 이름을 새로운 타입 이름으로 지정
        new_type_name = Counter(group).most_common(1)[0][0]
        for item in group:
            new_type_mapping[item] = new_type_name

# 새로운 타입 컬럼 추가
changwon_df['new_type'] = changwon_df['dog_facilities_type'].map(new_type_mapping)

# 동별 새로운 타입 개수 계산
summary_data = defaultdict(lambda: defaultdict(int))
for district, group in changwon_df.groupby('district'):
    type_counts = group['new_type'].value_counts()
    for new_type, count in type_counts.items():
        summary_data[district][new_type] = count

# 결과를 새로운 데이터 프레임으로 변환
summary_df = pd.DataFrame.from_dict(summary_data, orient='index').fillna(0).astype(int)
summary_df.index.name = 'district'
summary_df.reset_index(inplace=True)

# 출력 결과
print("새로운 타입이 추가된 원본 데이터프레임:")
print(changwon_df.head())

print("\n동별 새로운 타입 개수 요약 데이터프레임:")
print(summary_df)

changwon_df.to_csv("updated_data_c.csv", index=False)
summary_df.to_csv("summary_data_c.csv", index=False)
