# version01
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# 1. 데이터 불러오기
train_path = 'preprocessing_final_data_c.csv'
test_path = 'preprocessing_fianl_data_n.csv'

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# 2. 데이터셋 탐색
print("Train 데이터셋:")
print(train_data.info())
print(train_data.describe())

print("\nTest 데이터셋:")
print(test_data.info())
print(test_data.describe())

# 3. 독립변수(X)와 종속변수(y) 분리
X_train = train_data[["미용", "병원", "약국", "용품", "위탁"]]
y_train = train_data["견주수"]

X_test = test_data[["미용", "병원", "약국", "용품", "위탁"]]
y_test = test_data["견주수"]

# 4. 데이터 스케일링 (표준화)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. 다중공선성 확인 (VIF)
vif = pd.DataFrame()
vif["features"] = X_train.columns
vif["VIF Factor"] = [variance_inflation_factor(X_train_scaled, i) for i in range(X_train_scaled.shape[1])]
print("\n### 다중공선성 확인 (VIF) ###")
print(vif.round(2))

# 6. 선형 회귀 모델 학습
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 회귀 계수 확인
coefs = pd.DataFrame(zip(X_train.columns, model.coef_), columns=['feature', 'coefficients'])
print("\n### 회귀 계수 ###")
print(coefs.sort_values(by="coefficients", key=abs, ascending=False))

# 7. 모델 예측 및 평가
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# R^2 및 RMSE
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("\n### 모델 평가 ###")
print(f"Train R²: {train_r2:.3f}, Train RMSE: {train_rmse:.3f}")
print(f"Test R²: {test_r2:.3f}, Test RMSE: {test_rmse:.3f}")

# 8. 예측 결과 시각화
df_results = pd.DataFrame({'actual': y_test, 'prediction': y_pred_test})
df_results = df_results.sort_values(by='actual').reset_index(drop=True)

plt.figure(figsize=(12, 9))
plt.scatter(df_results.index, df_results['prediction'], marker='x', color='r', label='Prediction')
plt.scatter(df_results.index, df_results['actual'], alpha=0.6, color='black', label='Actual')
plt.title("Prediction Results")
plt.legend()
plt.show()

# 9. Statsmodels로 유의성 검정
X_train_const = sm.add_constant(X_train_scaled)
ols_model = sm.OLS(y_train, X_train_const).fit()
print(ols_model.summary())
###########
#version02
# 필요한 라이브러리 불러오기
import pandas as pd  # 데이터프레임 생성 및 데이터 조작
import numpy as np  # 수학적 계산 및 배열 작업
import matplotlib.pyplot as plt  # 시각화를 위한 라이브러리
from sklearn.linear_model import LinearRegression, SGDRegressor  # 선형 회귀 및 확률적 경사 하강법 회귀
from sklearn.preprocessing import StandardScaler  # 데이터 표준화 (스케일링)
from sklearn.metrics import mean_squared_error, r2_score  # 모델 평가를 위한 RMSE와 R^2 지표
from sklearn.model_selection import train_test_split, cross_val_score  # 데이터 분할 및 교차 검증
from statsmodels.stats.outliers_influence import variance_inflation_factor  # 다중공선성 확인
import statsmodels.api as sm  # 통계적 회귀 분석 라이브러리

# 1. 데이터 불러오기
# 학습 및 테스트 데이터셋 경로 설정
train_path = '/content/preprocessing_data_c.csv'  # 훈련 데이터 경로
test_path = '/content/preprocessing_fianl_data_n.csv'  # 테스트 데이터 경로

# CSV 파일을 pandas DataFrame으로 읽어옴
train_data = pd.read_csv(train_path)  # 훈련 데이터 불러오기
test_data = pd.read_csv(test_path)  # 테스트 데이터 불러오기

# 2. 독립변수(X)와 종속변수(y) 분리
# 훈련 데이터와 테스트 데이터에서 독립변수(입력)와 종속변수(타겟)를 분리
X_train = train_data[["미용", "병원", "약국", "용품", "위탁"]]  # 훈련 데이터의 독립변수 선택
y_train = train_data["견주수"]  # 훈련 데이터의 종속변수 선택
X_test = test_data[["미용", "병원", "약국", "용품", "위탁"]]  # 테스트 데이터의 독립변수 선택
y_test = test_data["견주수"]  # 테스트 데이터의 종속변수 선택

# 3. 데이터 스케일링
# 데이터의 분포를 표준화하여 모델 학습 성능을 높임
scaler = StandardScaler()  # StandardScaler 객체 생성
X_train_scaled = scaler.fit_transform(X_train)  # 훈련 데이터에 대해 스케일링 수행
X_test_scaled = scaler.transform(X_test)  # 동일한 스케일링 파라미터로 테스트 데이터 스케일링

# 4. 다중공선성 확인 (VIF: Variance Inflation Factor)
# 독립변수 간의 다중공선성 문제를 확인하기 위해 VIF 계산
vif = pd.DataFrame()  # VIF 결과를 저장할 DataFrame 생성
vif["features"] = X_train.columns  # 변수 이름 저장
# VIF 계산: 각 독립변수에 대해 다중공선성 계수를 계산
vif["VIF Factor"] = [variance_inflation_factor(X_train_scaled, i) for i in range(X_train_scaled.shape[1])]
print("\n### 다중공선성 확인 (VIF) ###")  # VIF 결과 출력
print(vif.round(2))  # 소수점 두 자리까지 반올림하여 출력

# 5. 모델 선택 및 학습
# 두 개의 회귀 모델(Linear Regression과 SGD Regressor)을 비교
models = {
    'Linear Regression': LinearRegression(),  # 선형 회귀 모델
    'SGD Regressor': SGDRegressor(max_iter=10000, tol=1e-6, random_state=42)  # 확률적 경사 하강법 회귀
}

# 모델 학습 및 평가를 저장할 딕셔너리 생성
results = {}

# 각 모델에 대해 반복
for name, model in models.items():
    model.fit(X_train_scaled, y_train)  # 모델 학습
    y_pred_train = model.predict(X_train_scaled)  # 훈련 데이터 예측
    y_pred_test = model.predict(X_test_scaled)  # 테스트 데이터 예측

    # 성능 평가 지표 계산
    train_r2 = r2_score(y_train, y_pred_train)  # 훈련 데이터 R² 점수
    test_r2 = r2_score(y_test, y_pred_test)  # 테스트 데이터 R² 점수
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))  # 훈련 데이터 RMSE
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))  # 테스트 데이터 RMSE

    # 교차 검증 수행
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')

    # 모델 평가 결과를 딕셔너리에 저장
    results[name] = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'cv_scores': cv_scores
    }

    # 결과 출력
    print(f"\n### {name} 모델 평가 ###")
    print(f"Train R²: {train_r2:.3f}, Train RMSE: {train_rmse:.3f}")
    print(f"Test R²: {test_r2:.3f}, Test RMSE: {test_rmse:.3f}")
    print(f"Cross-Validation R² scores: {cv_scores}")
    print(f"Average CV R²: {np.mean(cv_scores):.3f}")

# 6. 최적 모델 선택 및 시각화 (R² 기준)
# 테스트 데이터에서 R² 점수가 가장 높은 모델을 선택
best_model_name = max(results, key=lambda k: results[k]['test_r2'])  # R² 기준 최적 모델 이름
best_model = models[best_model_name]  # 최적 모델 불러오기
y_pred_test = best_model.predict(X_test_scaled)  # 최적 모델로 테스트 데이터 예측

# 실제 값과 예측 값을 비교하여 데이터프레임 생성
df_results = pd.DataFrame({'actual': y_test, 'prediction': y_pred_test})
df_results = df_results.sort_values(by='actual').reset_index(drop=True)  # 실제 값을 기준으로 정렬

# 예측 결과 시각화
plt.figure(figsize=(12, 9))  # 그래프 크기 설정
plt.scatter(df_results.index, df_results['prediction'], marker='x', color='r', label='Prediction')  # 예측 값 시각화
plt.scatter(df_results.index, df_results['actual'], alpha=0.6, color='black', label='Actual')  # 실제 값 시각화
plt.title(f"Prediction Results ({best_model_name})")  # 그래프 제목
plt.legend()  # 범례 표시
plt.show()  # 그래프 출력

# 7. Statsmodels를 이용한 회귀 분석 결과 출력
# 상수 항 추가 (절편을 위한 열 추가)
X_train_const = sm.add_constant(X_train_scaled)
ols_model = sm.OLS(y_train, X_train_const).fit()  # OLS(최소제곱법) 모델 학습
print(ols_model.summary())  # 회귀 분석 요약 결과 출력
###################
#version03
# 필요한 라이브러리 불러오기
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor  # 앙상블 모델 (Random Forest, Gradient Boosting)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor  # 선형 모델 및 확률적 경사 하강법 회귀
from sklearn.svm import SVR  # 서포트 벡터 회귀
from sklearn.neighbors import KNeighborsRegressor  # k-Nearest Neighbors 회귀
from sklearn.model_selection import GridSearchCV, cross_val_score  # 하이퍼파라미터 튜닝 및 교차 검증
from sklearn.pipeline import Pipeline  # 파이프라인 설정
from sklearn.preprocessing import StandardScaler  # 데이터 표준화
from sklearn.metrics import mean_squared_error, r2_score  # 성능 평가 지표 (RMSE, R²)
import pandas as pd  # 데이터 처리 및 조작
import numpy as np  # 수학적 계산 및 배열 처리
import matplotlib.pyplot as plt  # 시각화 라이브러리

# 1. 데이터 불러오기
# 훈련 및 테스트 데이터셋 경로 지정
train_path = '/content/preprocessing_data_c.csv'  # 훈련 데이터 경로
test_path = '/content/preprocessing_fianl_data_n.csv'  # 테스트 데이터 경로

# CSV 파일을 pandas DataFrame으로 읽어오기
train_data = pd.read_csv(train_path)  # 훈련 데이터 불러오기
test_data = pd.read_csv(test_path)  # 테스트 데이터 불러오기

# 2. 독립변수(X)와 종속변수(y) 분리
# 훈련 데이터에서 독립변수(입력 특성)와 종속변수(타겟)를 분리
X_train = train_data[["미용", "병원", "약국", "용품", "위탁"]]  # 독립변수 (X_train)
y_train = train_data["견주수"]  # 종속변수 (y_train)
X_test = test_data[["미용", "병원", "약국", "용품", "위탁"]]  # 테스트 독립변수 (X_test)
y_test = test_data["견주수"]  # 테스트 종속변수 (y_test)

# 3. 데이터 스케일링
# 특성 스케일링을 통해 데이터 분포를 표준화
scaler = StandardScaler()  # StandardScaler 객체 생성
X_train_scaled = scaler.fit_transform(X_train)  # 훈련 데이터 스케일링 (표준화)
X_test_scaled = scaler.transform(X_test)  # 테스트 데이터도 동일한 스케일로 변환

# 4. 모델 정의 및 하이퍼파라미터 튜닝
# 사용할 회귀 모델들을 딕셔너리에 정의
models = {
    'Linear Regression': LinearRegression(),  # 선형 회귀
    'Ridge': Ridge(),  # Ridge 회귀 (L2 규제)
    'Lasso': Lasso(),  # Lasso 회귀 (L1 규제)
    'ElasticNet': ElasticNet(),  # ElasticNet 회귀 (L1 + L2 규제)
    'SGD Regressor': SGDRegressor(max_iter=10000, tol=1e-6, random_state=42),  # SGD 회귀
    'SVR': SVR(),  # 서포트 벡터 회귀
    'Decision Tree': RandomForestRegressor(random_state=42),  # 의사결정 트리 (RandomForest 사용)
    'Random Forest': RandomForestRegressor(random_state=42),  # Random Forest 회귀
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),  # Gradient Boosting 회귀
    'kNN': KNeighborsRegressor()  # k-최근접 이웃 회귀
}

# Gradient Boosting 모델의 하이퍼파라미터 튜닝 설정
param_grid_gb = {
    'model__n_estimators': [100, 200, 300],  # 트리 개수
    'model__learning_rate': [0.01, 0.1, 0.2],  # 학습률
    'model__max_depth': [3, 5, 7]  # 트리의 최대 깊이
}

# SVR 모델의 하이퍼파라미터 튜닝 설정
param_grid_svr = {
    'model__kernel': ['linear', 'rbf'],  # 커널 종류
    'model__C': [1, 10, 100],  # 규제 강도
    'model__gamma': [0.01, 0.1, 1],  # 감마 값
    'model__epsilon': [0.1, 0.2, 0.5]  # 오차 허용 범위
}

# Random Forest 모델의 하이퍼파라미터 튜닝 설정
param_grid_rf = {
    'model__n_estimators': [100, 200, 300],  # 트리 개수
    'model__max_depth': [None, 10, 20],  # 최대 깊이
    'model__min_samples_split': [2, 5, 10],  # 분할을 위한 최소 샘플 수
    'model__min_samples_leaf': [1, 2, 4]  # 리프 노드에 필요한 최소 샘플 수
}

# 모델 평가 결과 저장을 위한 딕셔너리 생성
results = {}

# 모델 학습 및 평가 함수 정의
def run_model(name, model, param_grid=None):
    # 하이퍼파라미터 튜닝이 필요한 경우 GridSearchCV 적용
    if param_grid:
        pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])  # 파이프라인 설정
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)  # GridSearchCV 수행
        grid_search.fit(X_train, y_train)  # 모델 학습
        best_model = grid_search.best_estimator_  # 최적 모델 저장
    else:
        pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])  # 파이프라인 설정
        best_model = pipeline.fit(X_train, y_train)  # 모델 학습

    # 훈련 및 테스트 데이터 예측
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    # 성능 평가 지표 계산
    train_r2 = r2_score(y_train, y_pred_train)  # 훈련 데이터 R²
    test_r2 = r2_score(y_test, y_pred_test)  # 테스트 데이터 R²
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))  # 훈련 데이터 RMSE
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))  # 테스트 데이터 RMSE
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')  # 교차 검증 R² 점수

    # 결과 저장
    results[name] = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'cv_scores': cv_scores,
        'model': best_model
    }

    # 결과 출력
    print(f"\n### {name} 모델 평가 ###")
    print(f"Train R²: {train_r2:.3f}, Train RMSE: {train_rmse:.3f}")
    print(f"Test R²: {test_r2:.3f}, Test RMSE: {test_rmse:.3f}")
    print(f"Cross-Validation R² scores: {cv_scores}")
    print(f"Average CV R²: {np.mean(cv_scores):.3f}")

# 각 모델 평가 실행
for name, model in models.items():
    if name == 'Gradient Boosting':
        run_model(name, GradientBoostingRegressor(random_state=42), param_grid_gb)
    elif name == 'SVR':
        run_model(name, SVR(), param_grid_svr)
    elif name == 'Random Forest':
        run_model(name, RandomForestRegressor(random_state=42), param_grid_rf)
    else:
        run_model(name, model)

# 최적 모델 선택
best_model_name = max(results, key=lambda k: results[k]['test_r2'])  # R² 기준 최적 모델 선택
best_model = results[best_model_name]['model']  # 최적 모델 불러오기

# 최종 예측 결과 시각화
y_pred_test = best_model.predict(X_test)  # 테스트 데이터 예측
df_results = pd.DataFrame({'actual': y_test, 'prediction': y_pred_test})  # 실제 값과 예측 값을 DataFrame에 저장
df_results = df_results.sort_values(by='actual').reset_index(drop=True)  # 실제 값 기준 정렬

# 예측 결과 시각화
plt.figure(figsize=(12, 9))  # 그래프 크기 설정
plt.scatter(df_results.index, df_results['prediction'], marker='x', color='r', label='Prediction')  # 예측 값
plt.scatter(df_results.index, df_results['actual'], alpha=0.6, color='black', label='Actual')  # 실제 값
plt.title(f"Prediction Results ({best_model_name})")  # 그래프 제목
plt.legend()  # 범례
plt.show()  # 그래프 출력

#version04
# 필요한 라이브러리 불러오기
import pandas as pd  # 데이터 처리 및 분석을 위한 라이브러리
import numpy as np  # 수학적 계산 및 배열 처리를 위한 라이브러리
import matplotlib.pyplot as plt  # 데이터 시각화를 위한 라이브러리
from sklearn.model_selection import train_test_split, cross_val_score  # 데이터 분할 및 교차 검증
from sklearn.preprocessing import StandardScaler  # 데이터 스케일링 (표준화)
from sklearn.metrics import mean_squared_error, r2_score  # 모델 성능 평가 지표
from sklearn.linear_model import LinearRegression  # 선형회귀 모델

# 1. 데이터 불러오기
# 훈련 및 테스트 데이터셋 경로 지정
train_path = '/content/preprocessing_final_data_c.csv'  # 훈련 데이터 파일 경로
test_path = '/content/preprocessing_final_data_n.csv'  # 테스트 데이터 파일 경로

# CSV 파일을 pandas DataFrame으로 불러오기
train_data = pd.read_csv(train_path)  # 훈련 데이터 불러오기
test_data = pd.read_csv(test_path)  # 테스트 데이터 불러오기

# 2. 독립변수(X)와 종속변수(y) 분리
# 훈련 데이터에서 독립변수(특징)와 종속변수(타겟)를 분리
X_train = train_data[["미용", "병원", "약국", "용품", "위탁"]]  # 훈련 데이터의 독립변수
y_train = train_data["견주수"]  # 훈련 데이터의 종속변수
X_test = test_data[["미용", "병원", "약국", "용품", "위탁"]]  # 테스트 데이터의 독립변수
y_test = test_data["견주수"]  # 테스트 데이터의 종속변수

# 3. 데이터 스케일링
# 데이터의 분포를 표준화하여 모델의 학습 성능 향상
scaler = StandardScaler()  # StandardScaler 객체 생성 (평균 0, 표준편차 1로 변환)
X_train_scaled = scaler.fit_transform(X_train)  # 훈련 데이터 표준화
X_test_scaled = scaler.transform(X_test)  # 테스트 데이터도 같은 스케일로 변환

# 4. 선형회귀 모델 정의 및 학습
linear_model = LinearRegression()  # 선형회귀 모델 객체 생성
linear_model.fit(X_train_scaled, y_train)  # 훈련 데이터를 사용하여 모델 학습

# 5. 모델 예측
# 훈련 및 테스트 데이터셋에 대한 예측 수행
y_pred_train = linear_model.predict(X_train_scaled)  # 훈련 데이터 예측
y_pred_test = linear_model.predict(X_test_scaled)  # 테스트 데이터 예측

# 6. 모델 평가
# 성능 지표: R²(결정계수) 및 RMSE(평균 제곱근 오차) 계산
train_r2 = r2_score(y_train, y_pred_train)  # 훈련 데이터 R²
test_r2 = r2_score(y_test, y_pred_test)  # 테스트 데이터 R²
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))  # 훈련 데이터 RMSE
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))  # 테스트 데이터 RMSE

# 평가 결과 출력
print("\n### 선형회귀 모델 평가 ###")
print(f"Train R²: {train_r2:.3f}, Train RMSE: {train_rmse:.3f}")
print(f"Test R²: {test_r2:.3f}, Test RMSE: {test_rmse:.3f}")

# 7. 교차 검증 수행
# 교차 검증을 통해 모델의 일반화 성능을 확인
cv_scores = cross_val_score(linear_model, X_train_scaled, y_train, cv=5, scoring='r2')  # 5-폴드 교차 검증 수행

# 교차 검증 결과 출력
print("\n### 교차 검증 결과 ###")
print(f"Cross-Validation R² scores: {cv_scores}")
print(f"Average CV R²: {np.mean(cv_scores):.3f}")

# 8. 예측 결과 시각화
# 실제값과 예측값을 비교하여 시각화
df_results = pd.DataFrame({'actual': y_test, 'prediction': y_pred_test})  # 실제값과 예측값을 DataFrame에 저장
df_results = df_results.sort_values(by='actual').reset_index(drop=True)  # 실제값 기준으로 정렬

# 예측 결과 시각화
plt.figure(figsize=(12, 8))  # 그래프 크기 설정
plt.plot(df_results.index, df_results['prediction'], marker='x', linestyle='-', color='r', label='Prediction')  # 예측값
plt.plot(df_results.index, df_results['actual'], linestyle='-', alpha=0.6, color='black', label='Actual')  # 실제값
plt.title("Actual vs Predicted Results (Linear Regression)")  # 그래프 제목
plt.xlabel("Samples")  # x축 이름
plt.ylabel("견주수 (Owners Count)")  # y축 이름
plt.legend()  # 범례 표시
plt.show()  # 그래프 출력

##version05 딥러닝모델
# 필요한 라이브러리 불러오기
import pandas as pd  # 데이터 처리 및 분석
import numpy as np  # 수학적 계산 및 배열 처리
import matplotlib.pyplot as plt  # 시각화 라이브러리
from sklearn.model_selection import train_test_split, cross_val_score  # 데이터 분할 및 교차 검증
from sklearn.preprocessing import StandardScaler  # 데이터 표준화
from sklearn.metrics import mean_squared_error, r2_score  # 모델 평가 지표
from sklearn.linear_model import LinearRegression  # 선형회귀 모델
from tensorflow.keras.models import Sequential  # 딥러닝 모델 생성
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization  # 신경망 레이어
from tensorflow.keras.callbacks import EarlyStopping  # 학습 조기 종료

# 1. 데이터 불러오기
train_path = '/content/preprocessing_final_data_c.csv'  # 훈련 데이터 경로
test_path = '/content/preprocessing_final_data_n.csv'  # 테스트 데이터 경로

# CSV 파일 불러오기
train_data = pd.read_csv(train_path)  # 훈련 데이터
test_data = pd.read_csv(test_path)  # 테스트 데이터

# 2. 독립변수(X)와 종속변수(y) 분리
X_train = train_data[["미용", "병원", "약국", "용품", "위탁"]]  # 훈련 데이터의 독립변수
y_train = train_data["견주수"]  # 훈련 데이터의 종속변수
X_test = test_data[["미용", "병원", "약국", "용품", "위탁"]]  # 테스트 데이터의 독립변수
y_test = test_data["견주수"]  # 테스트 데이터의 종속변수

# 3. 데이터 스케일링
scaler = StandardScaler()  # StandardScaler 객체 생성
X_train_scaled = scaler.fit_transform(X_train)  # 훈련 데이터 표준화
X_test_scaled = scaler.transform(X_test)  # 테스트 데이터 표준화

# 4. 선형회귀 모델 학습 및 평가
# 선형회귀 모델 정의
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)  # 선형회귀 모델 학습

# 예측 수행
y_pred_train_lr = linear_model.predict(X_train_scaled)  # 훈련 데이터 예측
y_pred_test_lr = linear_model.predict(X_test_scaled)  # 테스트 데이터 예측

# 성능 평가
train_r2_lr = r2_score(y_train, y_pred_train_lr)
test_r2_lr = r2_score(y_test, y_pred_test_lr)
train_rmse_lr = np.sqrt(mean_squared_error(y_train, y_pred_train_lr))
test_rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_test_lr))

# 결과 출력
print("\n### 선형회귀 모델 평가 ###")
print(f"Train R²: {train_r2_lr:.3f}, Train RMSE: {train_rmse_lr:.3f}")
print(f"Test R²: {test_r2_lr:.3f}, Test RMSE: {test_rmse_lr:.3f}")

# 5. 딥러닝 모델 학습 및 평가
# 딥러닝 모델 정의
dl_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),  # 입력층 + 첫 번째 은닉층
    BatchNormalization(),  # 배치 정규화 (학습 안정화)
    Dropout(0.2),  # 과적합 방지를 위한 드롭아웃
    Dense(64, activation='relu'),  # 두 번째 은닉층
    Dropout(0.2),
    Dense(1, activation='linear')  # 출력층 (회귀 문제)
])

# 모델 컴파일
dl_model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # Adam 최적화와 MSE 손실함수 사용

# 학습 조기 종료 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 모델 학습
history = dl_model.fit(
    X_train_scaled, y_train,  # 학습 데이터
    validation_split=0.2,  # 검증 데이터 비율
    epochs=100,  # 학습 반복 횟수
    batch_size=16,  # 배치 크기
    callbacks=[early_stopping],  # 조기 종료 설정
    verbose=1  # 학습 과정 출력
)

# 딥러닝 모델 예측
y_pred_train_dl = dl_model.predict(X_train_scaled).flatten()  # 훈련 데이터 예측
y_pred_test_dl = dl_model.predict(X_test_scaled).flatten()  # 테스트 데이터 예측

# 성능 평가
train_r2_dl = r2_score(y_train, y_pred_train_dl)
test_r2_dl = r2_score(y_test, y_pred_test_dl)
train_rmse_dl = np.sqrt(mean_squared_error(y_train, y_pred_train_dl))
test_rmse_dl = np.sqrt(mean_squared_error(y_test, y_pred_test_dl))

# 결과 출력
print("\n### 딥러닝 모델 평가 ###")
print(f"Train R²: {train_r2_dl:.3f}, Train RMSE: {train_rmse_dl:.3f}")
print(f"Test R²: {test_r2_dl:.3f}, Test RMSE: {test_rmse_dl:.3f}")

# 6. 예측 결과 시각화 (딥러닝)
df_results = pd.DataFrame({'actual': y_test, 'prediction': y_pred_test_dl})
df_results = df_results.sort_values(by='actual').reset_index(drop=True)

plt.figure(figsize=(12, 8))
plt.scatter(df_results.index, df_results['prediction'], marker='x', color='r', label='Prediction')
plt.scatter(df_results.index, df_results['actual'], alpha=0.6, color='black', label='Actual')
plt.title("Prediction Results (Deep Learning)")
plt.xlabel("Samples")
plt.ylabel("견주수 (Owners Count)")
plt.legend()
plt.show()
