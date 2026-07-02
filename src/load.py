"""대용량 월별 CSV를 usecols+dtype 지정으로 로드하는 유틸."""
import pandas as pd

import config


def load_month(path, usecols=None):
    """CSV 파일 하나를 필요한 컬럼·dtype만 지정해 로드한다. usecols 미지정 시 전체 컬럼."""
    if usecols is None:
        usecols = config.COLUMNS

    dtype = {col: dt for col, dt in config.DTYPES.items() if col in usecols}
    parse_dates = ["event_time"] if "event_time" in usecols else False

    return pd.read_csv(path, usecols=usecols, dtype=dtype, parse_dates=parse_dates)


def load_months(paths, usecols=None):
    """여러 월 파일을 각각 load_month로 읽어 하나로 이어붙인다."""
    dfs = [load_month(path, usecols=usecols) for path in paths]
    return pd.concat(dfs, ignore_index=True)
