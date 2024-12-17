# 'Linear', 'Ridge', 'Lasso', 'SGD', 'Decision Tree','Random Forest','Gradient Boosting' 모델

# 필요한 라이브러리 불러오기
import pandas as pd  # 데이터 처리 및 조작
import numpy as np  # 수학적 계산 및 배열 처리
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor # 선형 모델
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor  # 앙상블 모델 (Random Forest, Gradient Boosting)
from sklearn.model_selection import GridSearchCV, cross_val_score  # 하이퍼파라미터 튜닝 및 교차 검증
from sklearn.pipeline import Pipeline  # 파이프라인 설정
from sklearn.preprocessing import StandardScaler  # 데이터 표준화
from sklearn.metrics import mean_squared_error, r2_score  # 성능 평가 지표 (RMSE, R²)
import seaborn as sns  # 시각화 라이브러리
import matplotlib.pyplot as plt  # 시각화 라이브러리

# 한글 설정
plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font',family='Malgun Gothic')

# 1. 데이터 불러오기
# 훈련 및 테스트 데이터셋 경로 지정
train_path = 'preprocessing_final_data_c.csv'
test_path = 'preprocessing_final_data_n.csv'
# 테스트 데이터 경로

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

# 3.1. 데이터 확인
# 막대그래프 시각화
train_data[["미용", "병원", "약국", "용품", "위탁"]].plot(kind="bar",figsize=(12,6))
plt.title("Bar Plot of Features and Target Variable")
plt.show()

# Boxplot 시각화
plt.figure(figsize=(10, 6))
sns.boxplot(data=train_data[["미용", "병원", "약국", "용품", "위탁"]])
plt.title("Boxplot of Features and Target Variable")
plt.show()

# heatmap 시각화
plt.figure(figsize=(10, 6))
sns.heatmap(train_data[["미용", "병원", "약국", "용품", "위탁", "견주수"]].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 4. 모델 정의 및 하이퍼파라미터 튜닝
# 사용할 회귀 모델들을 딕셔너리에 정의
models = {
    'Linear Regression': LinearRegression(),  # 선형 회귀
    'Ridge': Ridge(),  # Ridge 회귀 (L2 규제)
    'Lasso': Lasso(),  # Lasso 회귀 (L1 규제)
    'SGD Regressor': SGDRegressor(max_iter=10000, tol=1e-6, random_state=42),  # SGD 회귀
    'Decision Tree': RandomForestRegressor(random_state=42),  # 의사결정 트리 (RandomForest 사용)
    'Random Forest': RandomForestRegressor(random_state=42),  # Random Forest 회귀
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),  # Gradient Boosting 회귀
}

# Gradient Boosting 모델의 하이퍼파라미터 튜닝 설정
param_grid_gb = {
    'model__n_estimators': [100, 200, 300],  # 트리 개수
    'model__learning_rate': [0.01, 0.1, 0.2],  # 학습률
    'model__max_depth': [3, 5, 7]  # 트리의 최대 깊이
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

# 모델 학습 및 평가
for name, model in models.items():
    if name == 'Gradient Boosting':
        run_model(name, GradientBoostingRegressor(random_state=42), param_grid_gb)
    elif name == 'Random Forest':
        run_model(name, RandomForestRegressor(random_state=42), param_grid_rf)
    else:
        run_model(name, model)

# 5. 최적 모델 선택
best_model_name = max(results, key=lambda k: results[k]['test_r2'])  # R² 기준 최적 모델 선택
print("\n"+"최적 모델: ", best_model_name)
best_model = results[best_model_name]['model']  # 최적 모델 불러오기

# 6. 최종 예측 결과 시각화
y_pred_test = best_model.predict(X_test)  # 테스트 데이터 예측
df_results = pd.DataFrame({'actual': y_test, 'prediction': y_pred_test})  # 실제 값과 예측 값을 DataFrame에 저장
df_results = df_results.sort_values(by='actual').reset_index(drop=True)  # 실제 값 기준 정렬

# 예측 결과 시각화 (scatter plot)
plt.figure(figsize=(12, 9))  # 그래프 크기 설정
plt.scatter(df_results.index, df_results['prediction'], marker='x', color='r', label='Prediction')  # 예측 값
plt.scatter(df_results.index, df_results['actual'], alpha=0.6, color='black', label='Actual')  # 실제 값
plt.title(f"Prediction Results ({best_model_name})")  # 그래프 제목
plt.legend()  # 범례
plt.show()  # 그래프 출력

# 예측 결과 시각화 (line plot)
# 실제값과 예측값을 비교하여 시각화
df_results = pd.DataFrame({'actual': y_test, 'prediction': y_pred_test})  # 실제값과 예측값을 DataFrame에 저장
df_results = df_results.sort_values(by='actual').reset_index(drop=True)  # 실제값 기준으로 정렬

plt.figure(figsize=(12, 8))  # 그래프 크기 설정
plt.plot(df_results.index, df_results['prediction'], marker='x', linestyle='-', color='r', label='Prediction')  # 예측값
plt.plot(df_results.index, df_results['actual'], linestyle='-', alpha=0.6, color='black', label='Actual')  # 실제값
plt.title(f"Actual vs Predicted Results ({best_model_name})")  # 그래프 제목
plt.xlabel("Samples")  # x축 이름
plt.ylabel("견주수 (Owners Count)")  # y축 이름
plt.legend()  # 범례 표시
plt.show()  # 그래프 출력
