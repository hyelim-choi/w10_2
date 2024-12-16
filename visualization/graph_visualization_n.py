import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 설정

c_df = pd.read_csv("../preprocessing_all/preprocessing_fianl_data_n.csv")  # 남양주 데이터 불러오기

# 서브플롯 설정
f, axes = plt.subplots(2, 3)
f.set_size_inches((20, 15))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

c_df["all_facilities"] = c_df["미용"]+c_df["병원"]+c_df["약국"]+c_df["용품"]+c_df["위탁"]  # 전체 시설 열 추가

c_df = c_df.sort_values(by=["all_facilities"], ascending=False)  # 시설 개수로 정렬
# bar plot 쌓기
axes[0, 0].bar(c_df["동"], c_df["미용"], label="beauty", color="red", alpha=0.3)
axes[0, 0].bar(c_df["동"], c_df["병원"], label="hospital", bottom=c_df["미용"], color="orange", alpha=0.3)
axes[0, 0].bar(c_df["동"], c_df["약국"], label="pharmacy", bottom=c_df["미용"] + c_df["병원"], color="yellow", alpha=0.3)
axes[0, 0].bar(c_df["동"], c_df["용품"], label="product", bottom=c_df["미용"] + c_df["병원"] + c_df["약국"], color="green", alpha=0.3)
axes[0, 0].bar(c_df["동"], c_df["위탁"], label="foster_place", bottom=c_df["미용"] + c_df["병원"] + c_df["약국"] + c_df["용품"], color="purple", alpha=0.3)
axes[0, 0].plot(c_df["동"], c_df["견주수"]*0.01, label="ratio", color="black", alpha=0.7)  # 견주수 막대그래프 (비교를 위해 스케일 조정)
axes[0, 0].set_title("반려견 시설(전체)별 반려인 수")  # 제목 설정
axes[0, 0].legend()  # 범례 표시

c_df = c_df.sort_values(by=["미용"], ascending=False)  # 미용 개수로 정렬
axes[0, 1].bar(c_df["동"], c_df["미용"], label="beauty", color="red", alpha=0.3)  # bar plot
axes[0, 1].plot(c_df["동"], c_df["견주수"]*0.005, label="ratio", color="black", alpha=0.7)  # 견주수 막대그래프 (비교를 위해 스케일 조정)
axes[0, 1].set_title("반려견 시설(미용)별 반려인 수")  # 제목 설정
axes[0, 1].legend()  # 범례 표시

c_df = c_df.sort_values(by=["병원"], ascending=False)  # 병원 개수로 정렬
axes[0, 2].bar(c_df["동"], c_df["병원"], label="hospital", color="orange", alpha=0.3)  # bar plot
axes[0, 2].plot(c_df["동"], c_df["견주수"]*0.005, label="ratio", color="black", alpha=0.7)  # 견주수 막대그래프 (비교를 위해 스케일 조정)
axes[0, 2].set_title("반려견 시설(병원)별 반려인 수")  # 제목 설정
axes[0, 2].legend()  # 범례 표시

c_df = c_df.sort_values(by=["약국"], ascending=False)  # 약국 개수로 정렬
axes[1, 0].bar(c_df["동"], c_df["약국"], label="pharmacy", color="yellow", alpha=0.3)  # bar plot
axes[1, 0].plot(c_df["동"], c_df["견주수"]*0.005, label="ratio", color="black", alpha=0.7)  # 견주수 막대그래프 (비교를 위해 스케일 조정)
axes[1, 0].set_title("반려견 시설(약국)별 반려인 수")  # 제목 설정
axes[1, 0].legend()  # 범례 표시

c_df = c_df.sort_values(by=["용품"], ascending=False)  # 용품 개수로 정렬
axes[1, 1].bar(c_df["동"], c_df["용품"], label="product", color="green", alpha=0.3)  # bar plot
axes[1, 1].plot(c_df["동"], c_df["견주수"]*0.005, label="ratio", color="black", alpha=0.7)  # 견주수 막대그래프 (비교를 위해 스케일 조정)
axes[1, 1].set_title("반려견 시설(용품)별 반려인 수")  # 제목 설정
axes[1, 1].legend()  # 범례 표시

c_df = c_df.sort_values(by=["위탁"], ascending=False)  # 위탁 개수로 정렬
axes[1, 2].bar(c_df["동"], c_df["위탁"], label="foster_place", color="purple", alpha=0.3)  # bar plot
axes[1, 2].plot(c_df["동"], c_df["견주수"]*0.005, label="ratio", color="black", alpha=0.7)  # 견주수 막대그래프 (비교를 위해 스케일 조정)
axes[1, 2].set_title("반려견 시설(위탁)별 반려인 수")  # 제목 설정
axes[1, 2].legend()  # 범례 표시

for ax in axes.flat:
    ax.axes.xaxis.set_visible(False)  # x축 제거
    ax.axes.yaxis.set_visible(False)  # y축 제거

plt.show()  # 그래프 확인
