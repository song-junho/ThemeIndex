
import pandas as pd

import pickle


from tqdm import tqdm
from functools import reduce
from collections import deque
import warnings
import threading
from concurrent.futures import ThreadPoolExecutor, wait

class Theme:

    def __init__(self, date_start, date_end):

        self.list_date_eos = pd.date_range(start="2006-01-01", end="2023-06-01", freq="MS")
        self.list_date_eom = pd.date_range(start="2006-01-01", end="2023-06-01", freq="M")

        # 테마 인덱스
        self.dict_theme_index = {}

        # 테마 인덱스 (월별 구간 수익률)
        self.dict_monthly_theme_index = {}

        # 테마별 종목 가격 데이터
        self.dict_theme_cmp = {}

        # Thread Lock
        self._lock = threading.Lock()

        # 가격 Dictionary 생성
        with open(r"D:\MyProject\StockPrice\DictDfStock.pickle", 'rb') as fr:
            self.dict_df_stock = pickle.load(fr)

        # 테마 키워드
        self.df_cmp_keyword = pd.read_excel(r"D:\MyProject\Notion\키워드_사전.xlsx", dtype="str")

    def insert_monthly_theme_index(self, list_cmp_cd, theme, eos, eom):

        monthly_index = deque([])

        for cmp_cd in list_cmp_cd:
            df_stock = self.dict_theme_cmp[theme][cmp_cd][["Close"]]
            df_stock = df_stock.loc[eos:eom]

            # 당월 데이터가 15영업일 미만인 종목은 제외
            if len(df_stock) < 15:
                continue
            else:
                df_stock["Close"] = df_stock["Close"] / df_stock["Close"].iloc[0]

            monthly_index.append(df_stock)

        if len(monthly_index) == 0:
            return
        else:
            df = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), monthly_index)
            df["index"] = df.mean(axis='columns')
            self.dict_monthly_theme_index[theme].append(df[["index"]])

    def thread_theme(self, list_theme, eos, eom):

        # 테마 인덱싱 스레드 함수
        for theme in list_theme:
            self.insert_monthly_theme_index(self.dict_theme_cmp[theme].keys(), theme, eos, eom)

    def make_theme_index(self):

        # 월별 인덱스 생성
        for eos, eom in tqdm(zip(self.list_date_eos, self.list_date_eom), total=len(self.list_date_eos)):

            n = 100
            list_theme_t = sorted(self.dict_theme_cmp.keys())
            list_theme_t = [list_theme_t[i * n:(i + 1) * n] for i in range((len(list_theme_t) + n - 1) // n)]

            threads = []
            with ThreadPoolExecutor(max_workers=5) as executor:

                for list_theme in list_theme_t:
                    threads.append(executor.submit(self.thread_theme, list_theme, eos, eom))
                wait(threads)

    def index_strapping(self):

        for theme_keyword in tqdm(self.dict_monthly_theme_index.keys()):

            monthly_theme = self.dict_monthly_theme_index[theme_keyword]

            df_theme_index = pd.DataFrame()
            for month_index in monthly_theme:

                # 직전 인덱스 값 초기화
                if len(df_theme_index) == 0:
                    latest_index = 1
                else:
                    latest_index = df_theme_index.iloc[-1]["index"]

                df = pd.DataFrame(month_index["index"] * latest_index)
                df_theme_index = pd.concat([df_theme_index, df])

            self.dict_theme_index[theme_keyword] = df_theme_index

    def create_theme_index(self):

        # 인덱스 생성용 키워드 선별
        df = self.df_cmp_keyword.groupby("keyword").count()
        df = (df[df["cmp_cd"] >= 5])
        list_theme = list(df[df["cmp_cd"] > 4].index)

        df_theme = self.df_cmp_keyword[self.df_cmp_keyword["keyword"].isin(list_theme)]

        # 테마 내 종목별 가격데이터 생성
        for theme in tqdm(list_theme):

            list_cmp_cd = df_theme[df_theme["keyword"] == theme]["cmp_cd"].to_list()
            self.dict_theme_cmp[theme] = {}

            for cmp_cd in list_cmp_cd:
                self.dict_theme_cmp[theme][cmp_cd] = self.dict_df_stock[cmp_cd]

        # 테마 월별 구간 수익률 적재용 , dict_monthly_theme_index 초기화
        for theme in (self.dict_theme_cmp.keys()):
            self.dict_monthly_theme_index[theme] = deque([])

        # 테마 인덱스 생성
        self.make_theme_index()

        # 월별 구간 수익률 strapping
        self.index_strapping()

        # 저장
        with open(r'D:\MyProject\StockPrice\DictThemeIndex.pickle', 'wb') as fw:
            pickle.dump(self.dict_theme_index, fw)
