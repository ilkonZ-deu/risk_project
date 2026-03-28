import pandas as pd
import numpy as np
import datetime

def find_stock_returns(df_stocks: pd.DataFrame) -> pd.DataFrame:
    """
    Найти доходности активов (акций) по дням

    Args:
        df_stocks (pandas.DataFrame): датафрейм с ценами акций
    """
    df_diff = df_stocks - df_stocks.shift(1)
    df_returns = df_diff / df_stocks
    df_returns = df_returns.dropna()
    return df_returns
from typing import Literal

def find_market_portfolio_returns(df_returns: pd.DataFrame, return_type: Literal['series', 'dataframe']='series'):
    """
    Найти доходность рыночного портфеля в прошлом
    """
    series_market = df_returns.mean(axis=1)
    if return_type == 'series':
        return series_market
    df_market = pd.DataFrame(series_market, columns=['MOEX30'])
    return df_market
from typing import Literal

def estimate_betas(df_returns: pd.DataFrame, start_month: str, end_month: str, method: Literal['historical', 'adjusted']='historical'):
    if method not in ['historical', 'adjusted']:
        raise ValueError('Unsupported beta estimation method: {0}'.format(method))
    df = df_returns.copy()
    df = df.loc[start_month:end_month]
    df['MOEX30'] = find_market_portfolio_returns(df, 'series')
    df_cov = df.cov().filter(['MOEX30']).drop(index=['MOEX30'])
    market_var = df.var(axis=0).get('MOEX30')
    df_beta = df_cov / market_var
    df_beta = df_beta.rename(columns={'MOEX30': 'beta'})
    if method == 'historical':
        return df_beta
    alpha0 = 0.333
    alpha1 = 0.666
    df_beta['beta'] = alpha0 + alpha1 * df_beta['beta']
    return df_beta

def find_beta_cov_matrix(df_returns: pd.DataFrame, start_month: str, end_month: str, beta_method: str='historical'):
    df = df_returns.copy()
    df_beta = estimate_betas(df, start_month, end_month, method=beta_method)
    market_var = find_market_portfolio_returns(df, return_type='series').var()
    df_var = pd.DataFrame(market_var * np.outer(df_beta.values, df_beta.values), index=df_beta.index, columns=df_beta.index)
    return df_var

def find_historical_cov_matrix(df_returns: pd.DataFrame, start_month: str, end_month: str):
    df = df_returns.loc[start_month:end_month]
    df = df.copy()
    return df.cov()
from typing import Literal

def find_returns_cov(df_returns, start_month: str, end_month: str, method: Literal['historical', 'beta_historical', 'beta_adjusted']):
    if method == 'historical':
        return find_historical_cov_matrix(df_returns, start_month, end_month)
    elif method == 'beta_historical':
        return find_beta_cov_matrix(df_returns, start_month, end_month, beta_method='historical')
    elif method == 'beta_adjusted':
        return find_beta_cov_matrix(df_returns, start_month, end_month, beta_method='adjusted')
    raise ValueError('Unsupported method: {0}'.format(method))

def find_returns_mu(df_returns: pd.DataFrame, start_month: str, end_month: str):
    df = df_returns.loc[start_month:end_month]
    df = pd.DataFrame(df.mean(axis=0), columns=['mu'])
    return df

def portfolio_return(weights, mu):
    return np.dot(weights, mu)

def portfolio_volatility(weights, cov):
    return np.sqrt(weights @ cov @ weights)

def portfolio_sharpe(weights, mu, cov):
    ret = portfolio_return(weights, mu)
    vol = portfolio_volatility(weights, cov)
    return ret / vol if vol > 0 else 0
from scipy.optimize import minimize

def find_effective_frontier(df_returns: pd.DataFrame, start_month: str, end_month: str, cov_method: Literal['historical', 'beta_historical', 'beta_adjusted']):
    mu = df_returns.loc[start_month:end_month].mean(axis=0)
    mu = mu.values
    cov = find_returns_cov(df_returns, start_month, end_month, method=cov_method)
    cov = cov.values
    min_return = np.min(mu)
    max_return = np.max(mu)
    returns_arr = np.linspace(min_return, max_return, 50)
    n_assets = mu.size
    w0 = np.full(n_assets, fill_value=1 / n_assets)
    frontier_returns = []
    frontier_volatilities = []
    for target in returns_arr:
        result = minimize(lambda w: portfolio_volatility(w, cov), x0=w0, bounds=[(0, None) for _ in w0], constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}, {'type': 'eq', 'fun': lambda w: portfolio_return(w, mu) - target}], options={'maxiter': 1000})
        if not result.success:
            continue
        w = result.x
        frontier_returns.append(portfolio_return(w, mu))
        frontier_volatilities.append(portfolio_volatility(w, cov))
    return (frontier_volatilities, frontier_returns)
import requests
import pandas as pd
import time

def get_moex_candles(ticker, start_date, end_date, interval=24):
    """
    Скачивает дневные свечи по одной акции с MOEX ISS
    """
    url = f'{BASE_URL}/engines/stock/markets/shares/securities/{ticker}/candles.json'
    all_rows = []
    start = 0
    while True:
        params = {'from': start_date, 'till': end_date, 'interval': interval, 'start': start}
        response = session.get(url, params=params, timeout=30)
        response.raise_for_status()
        j = response.json()
        candles = j.get('candles', {})
        columns = candles.get('columns', [])
        data = candles.get('data', [])
        if not data:
            break
        chunk = pd.DataFrame(data, columns=columns)
        chunk['ticker'] = ticker
        all_rows.append(chunk)
        if len(data) < 100:
            break
        start += len(data)
        time.sleep(0.2)
    if not all_rows:
        return pd.DataFrame()
    df = pd.concat(all_rows, ignore_index=True)
    return df

def load_all_tickers(tickers, start_date, end_date):
    """
    Скачивает данные по списку тикеров и собирает в одну таблицу
    """
    frames = []
    for i, ticker in enumerate(tickers, 1):
        print(f'[{i}/{len(tickers)}] Загружаю {ticker}...')
        try:
            df = get_moex_candles(ticker, start_date, end_date)
            if df.empty:
                print(f'  Нет данных по {ticker}')
                continue
            frames.append(df)
        except Exception as e:
            print(f'  Ошибка для {ticker}: {e}')
        time.sleep(0.3)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    return result

def get_rebalance_dates(index, step='Y'):
    """
    Выбирает даты, на которых считаем mu и Sigma.
    'Y'  - конец года
    'Q'  - конец квартала
    'M'  - конец месяца
    'W'  - конец недели
    'D'  - каждый день
    """
    idx = pd.DatetimeIndex(index)
    if step == 'D':
        return idx
    s = pd.Series(index=idx, data=1)
    dates = s.resample(step).last().dropna().index
    return dates

def mean_cov_unweighted(window_returns):
    """
    window_returns: DataFrame (T x N)
    Возвращает:
        mu    : Series (N,)
        Sigma : DataFrame (N x N)
    """
    mu = window_returns.mean()
    Sigma = window_returns.cov()
    return (mu, Sigma)

def exp_weights(n, lam=0.94):
    """
    n - число наблюдений
    lam - коэффициент забывания
    Вес самого свежего наблюдения максимален
    """
    w = np.array([lam ** k for k in range(n - 1, -1, -1)], dtype=float)
    w /= w.sum()
    return w

def mean_cov_ew(window_returns, lam=0.94):
    """
    Экспоненциально-взвешенные mean и covariance
    """
    X = window_returns.values
    n, m = X.shape
    w = exp_weights(n, lam=lam)
    mu = np.sum(X * w[:, None], axis=0)
    X_centered = X - mu
    Sigma = (X_centered * w[:, None]).T @ X_centered
    mu = pd.Series(mu, index=window_returns.columns, name='mu')
    Sigma = pd.DataFrame(Sigma, index=window_returns.columns, columns=window_returns.columns)
    return (mu, Sigma)

def rolling_mean_cov(returns, window_days=252, step='Y', weighted=False, lam=0.94):
    rebalance_dates = get_rebalance_dates(returns.index, step=step)
    mus = {}
    covs = {}
    for dt in rebalance_dates:
        sub = returns.loc[:dt]
        if len(sub) < window_days:
            continue
        window = sub.iloc[-window_days:]
        if weighted:
            mu, Sigma = mean_cov_ew(window, lam=lam)
        else:
            mu, Sigma = mean_cov_unweighted(window)
        mus[dt] = mu
        covs[dt] = Sigma
    return (mus, covs)

def expanding_mean_cov(returns, min_days=252, step='Y', weighted=False, lam=0.94):
    """
    expanding window: от начала выборки до даты dt
    min_days: минимальное число наблюдений перед первым расчетом
    """
    rebalance_dates = get_rebalance_dates(returns.index, step=step)
    mus = {}
    covs = {}
    for dt in rebalance_dates:
        window = returns.loc[:dt]
        if len(window) < min_days:
            continue
        if weighted:
            mu, Sigma = mean_cov_ew(window, lam=lam)
        else:
            mu, Sigma = mean_cov_unweighted(window)
        mus[dt] = mu
        covs[dt] = Sigma
    return (mus, covs)