
import pandas as pd
import pickle

from tqdm import tqdm
from functools import reduce
from collections import deque
import warnings
import threading
from concurrent.futures import ThreadPoolExecutor, wait
import copy
from functools import reduce
import exchange_calendars as ecals
import gc


import warnings
warnings.filterwarnings('ignore')

# 한국장 영업일
XKRX = ecals.get_calendar("XKRX")

class ThemeIndex:

    def __init__(self, start_date, end_date):

        self.list_date_som = pd.date_range(start=start_date, end=end_date, freq="MS")
        self.list_date_eom = pd.date_range(start=start_date, end=end_date, freq="M")

        # 테마 인덱스
        self.dict_theme_index = {}

        # 테마 인덱스 (월별 구간 수익률)
        self.dict_monthly_theme_index = {}

        # 테마별 종목 가격 데이터
        self.dict_theme_cmp = {}

        # 한국장 영업일
        self.list_krx_date = XKRX.schedule.index

        # Thread Lock
        self._lock = threading.Lock()

        # 가격 Dictionary 생성
        with open(r"D:\MyProject\StockPrice\DictDfStock.pickle", 'rb') as fr:
            self.dict_df_stock = pickle.load(fr)

        # 테마 키워드
        self.df_cmp_keyword = pd.read_excel(r"D:\MyProject\Notion\키워드_사전.xlsx", dtype="str")

    def insert_monthly_theme_index(self, list_cmp_cd, theme, som, eom):

        monthly_index = deque([])

        limit_len = len(list(filter(lambda x: x if (x > som) & (x < eom) else None, XKRX.schedule.index)))

        for cmp_cd in list_cmp_cd:
            df_stock = self.dict_theme_cmp[theme][cmp_cd]
            df_stock = df_stock.loc[som:eom]
            df_stock = df_stock[df_stock["Volume"] > 0] # 거래 정지 구간 제외

            # 당월 데이터가 삼성전자 영업일*0.8 미만인 종목은 제외
            if len(df_stock) < limit_len:
                continue
            else:
                df_stock["Close"] = df_stock["Close"] / df_stock["Close"].iloc[0]

            monthly_index.append(df_stock[["Close"]])

        if len(monthly_index) == 0:
            return
        else:
            df = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), monthly_index)
            df["index"] = df.mean(axis='columns')
            self.dict_monthly_theme_index[theme].append(df[["index"]])

    def thread_theme(self, list_theme, som, eom):

        # 테마 인덱싱 스레드 함수
        for theme in list_theme:
            self.insert_monthly_theme_index(self.dict_theme_cmp[theme].keys(), theme, som, eom)

    def make_theme_index(self):

        # 월별 인덱스 생성
        for som, eom in tqdm(zip(self.list_date_som, self.list_date_eom), total=len(self.list_date_som)):

            n = 100
            list_theme_t = sorted(self.dict_theme_cmp.keys())
            list_theme_t = [list_theme_t[i * n:(i + 1) * n] for i in range((len(list_theme_t) + n - 1) // n)]

            threads = []
            with ThreadPoolExecutor(max_workers=5) as executor:

                for list_theme in list_theme_t:
                    threads.append(executor.submit(self.thread_theme, list_theme, som, eom))
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

        del self.dict_df_stock
        gc.collect()

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


class ThemeChgFreq:

    '''
    테마 월간 데이터 , 1M, 3M, 6M, 12M 주기별 변화율 저장
    '''

    def __init__(self):

        with open(r'D:\MyProject\StockPrice\DictThemeIndex.pickle', 'rb') as fr:
            self.dict_theme_index = pickle.load(fr)

        self.dict_theme_chg_freq = {}


    def preprocessing(self, df, val_nm):

        df = df.resample("1M").last()
        df = df.reset_index(drop=False)
        df["pct_change"] = df[val_nm].pct_change()
        df["Date"] = pd.to_datetime(df["Date"].dt.strftime("%Y-%m-%d"))
        df = df[["Date", val_nm, "pct_change"]].rename(columns={"Date": "date", val_nm: "val"})
        df = df.set_index("date")

        return df

    def save(self):

        with open(r'D:\MyProject\StockPrice\DictThemeChgFreq.pickle', 'wb') as fw:
            pickle.dump(self.dict_theme_chg_freq, fw)

    def create_chg_freq(self):

        list_theme_nm = list(self.dict_theme_index.keys())

        # 전처리: 월간 데이터 형식
        for key_nm in tqdm(list_theme_nm):
            df = self.dict_theme_index[key_nm]
            self.dict_theme_index[key_nm] = self.preprocessing(df, "index")

        # 주기별 변화율 데이터 생성
        for key_nm in tqdm(list_theme_nm):

            list_tmp = []
            for freq in [1, 3, 6, 12]:
                df = copy.deepcopy(self.dict_theme_index[key_nm])
                if freq == 1:
                    df = df[["pct_change"]].rename(columns={"pct_change": str(freq) + "M"})
                    list_tmp.append(df)
                else:
                    df["pct_change"] = df["val"].pct_change(freq)
                    df = df[["pct_change"]].rename(columns={"pct_change": str(freq) + "M"})
                    list_tmp.append(df)

            df_chg = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), list_tmp)
            df = pd.DataFrame(df_chg.T.isnull().sum())
            not_nan_date = min(df[df[0] == 0].index)
            df_chg = df_chg.loc[not_nan_date:]

            self.dict_theme_chg_freq[key_nm] = pd.merge(left=self.dict_theme_index[key_nm][["val"]], right=df_chg, left_index=True,
                                                right_index=True)

        self.save()