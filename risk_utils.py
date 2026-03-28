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


def efficient_frontier_unconstrained(mu_vec, cov_mat, n_points=80):
    """
    Граница минимальной дисперсии при целевой доходности без ограничений на веса
    (короткие продажи разрешены, sum w = 1). Возвращает (vols, rets) или (None, None).
    """
    mu_vec = np.asarray(mu_vec, dtype=float).ravel()
    cov_mat = np.asarray(cov_mat, dtype=float)
    n = len(mu_vec)
    x0 = np.ones(n) / n
    constraints_sum = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

    res_min = minimize(
        lambda w: portfolio_volatility(w, cov_mat),
        x0,
        constraints=constraints_sum,
        method='SLSQP',
        options={'maxiter': 500, 'ftol': 1e-10},
    )
    if not res_min.success:
        return None, None
    ret_min = portfolio_return(res_min.x, mu_vec)
    ret_max = float(np.max(mu_vec)) * 1.5
    target_rets = np.linspace(ret_min, ret_max, n_points)
    vols, rets = [], []
    for target in target_rets:
        cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'eq', 'fun': lambda w, tr=target: portfolio_return(w, mu_vec) - tr},
        ]
        res = minimize(
            lambda w: portfolio_volatility(w, cov_mat),
            x0,
            constraints=cons,
            method='SLSQP',
            options={'maxiter': 500, 'ftol': 1e-10},
        )
        if res.success:
            vols.append(portfolio_volatility(res.x, cov_mat))
            rets.append(portfolio_return(res.x, mu_vec))
    return np.array(vols), np.array(rets)


def maximum_variance_frontier_long_only(mu_vec, cov_mat, n_points=80):
    """
    Верхняя граница (максимальная дисперсия) при целевой доходности, long-only:
    w >= 0, sum w = 1. Возвращает (vols, rets, weights) — массивы или пустые списки при сбое.
    """
    mu_vec = np.asarray(mu_vec, dtype=float).ravel()
    cov_mat = np.asarray(cov_mat, dtype=float)
    n = len(mu_vec)
    x0 = np.ones(n) / n
    bounds = [(0.0, None) for _ in range(n)]
    target_rets = np.linspace(float(np.min(mu_vec)), float(np.max(mu_vec)), n_points)
    vols, rets, weights = [], [], []
    for target in target_rets:
        cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'eq', 'fun': lambda w, tr=target: portfolio_return(w, mu_vec) - tr},
        ]
        res = minimize(
            lambda w: -portfolio_volatility(w, cov_mat),
            x0,
            bounds=bounds,
            constraints=cons,
            method='SLSQP',
            options={'maxiter': 500, 'ftol': 1e-9},
        )
        if res.success:
            w = res.x
            vols.append(portfolio_volatility(w, cov_mat))
            rets.append(portfolio_return(w, mu_vec))
            weights.append(w.copy())
            x0 = w.copy()
    if not vols:
        return np.array([]), np.array([]), []
    return np.array(vols), np.array(rets), weights


def _random_feasible_wu(rng, n, gross_leverage):
    """Случайное (w, u): sum w=1, sum u <= L, u_i >= |w_i|."""
    L = float(gross_leverage)
    w = rng.random(n)
    w = w / np.sum(w)
    u = np.abs(w) + rng.random(n) * 0.05 + 1e-4
    u = np.maximum(u, np.abs(w))
    if np.sum(u) > L:
        u = u * (L / np.sum(u))
    return np.concatenate([w, u])


def maximum_variance_frontier_gross_leverage(
    mu_vec, cov_mat, gross_leverage=2.0, n_points=80, n_starts=2, rng=None
):
    """
    Максимизация дисперсии при целевой доходности с разрешёнными шортами и ограничением
    валовой экспозиции sum_i |w_i| <= gross_leverage. Вспомогательные u_i >= |w_i|:
    sum u <= L, u_i - w_i >= 0, u_i + w_i >= 0. Также sum w = 1, w'mu = r.

    Возвращает (vols, rets, weights_list). При неуспехе всех стартов точка пропускается.
    """
    mu_vec = np.asarray(mu_vec, dtype=float).ravel()
    cov_mat = np.asarray(cov_mat, dtype=float)
    n = len(mu_vec)
    L = float(gross_leverage)
    if L < 1.0:
        raise ValueError('gross_leverage must be >= 1 (need at least sum|w| >= |sum w| = 1)')
    rng = np.random.default_rng(42) if rng is None else rng

    def w_from_z(z):
        return z[:n]

    def neg_var(z):
        w = w_from_z(z)
        return -float(w @ cov_mat @ w)

    def make_constraints(target_r):
        return [
            {'type': 'eq', 'fun': lambda z, tr=target_r: np.sum(z[:n]) - 1.0},
            {'type': 'eq', 'fun': lambda z, tr=target_r: portfolio_return(w_from_z(z), mu_vec) - tr},
            {'type': 'ineq', 'fun': lambda z: L - np.sum(z[n:])},
            {'type': 'ineq', 'fun': lambda z: z[n:] - z[:n]},
            {'type': 'ineq', 'fun': lambda z: z[n:] + z[:n]},
        ]

    bounds = [(None, None)] * n + [(0.0, None)] * n
    target_rets = np.linspace(float(np.min(mu_vec)), float(np.max(mu_vec)), n_points)
    vols, rets, weights = [], [], []

    w0 = np.ones(n) / n
    u0 = np.abs(w0) + 1e-4
    if np.sum(u0) > L:
        u0 = u0 * (L / np.sum(u0))
    x0_det = np.concatenate([w0, u0])

    for target in target_rets:
        best_w = None
        best_var = -np.inf
        starts = [x0_det.copy()]
        for _ in range(max(0, n_starts - 1)):
            starts.append(_random_feasible_wu(rng, n, L))

        for x0 in starts:
            res = minimize(
                neg_var,
                x0,
                bounds=bounds,
                constraints=make_constraints(target),
                method='SLSQP',
                options={'maxiter': 400, 'ftol': 1e-7},
            )
            if res.success:
                w = w_from_z(res.x)
                v = float(w @ cov_mat @ w)
                if v > best_var:
                    best_var = v
                    best_w = w
        if best_w is not None:
            vols.append(np.sqrt(max(best_var, 0.0)))
            rets.append(portfolio_return(best_w, mu_vec))
            weights.append(best_w.copy())
            u_new = np.maximum(np.abs(best_w), 1e-8)
            if np.sum(u_new) > L:
                u_new = u_new * (L / np.sum(u_new))
            x0_det = np.concatenate([best_w, u_new])

    if not vols:
        return np.array([]), np.array([]), []
    return np.array(vols), np.array(rets), weights


def adv_dollar_proxy_from_volatility(cov_mat, median_scale=0.05):
    """
    Прокси среднего дневного объёма в долях капитала: из волатильностей sqrt(diag Sigma).
    Выше sigma_i — ниже относительная «глубина», ADV масштабируется медианой.
    """
    cov_mat = np.asarray(cov_mat, dtype=float)
    sig = np.sqrt(np.maximum(np.diag(cov_mat), 1e-18))
    inv = 1.0 / (sig + 1e-8)
    med = np.median(inv)
    if med <= 0:
        med = 1.0
    adv = inv / med * float(median_scale)
    return adv


def expected_execution_impact_ac(
    w,
    w_prev,
    cov_mat,
    adv_dollar=None,
    alpha=0.5,
    eta_temp=1.0,
    eta_perm=0.25,
    linear_bps=5.0,
    notional=1.0,
    participation_cap=1.0,
    eps=1e-12,
):
    """
    Reduced-form Almgren–Chriss / square-root: temporary ~ participation**alpha,
    permanent ~ participation, participation_i = |dw_i|*notional / ADV_i.
    Без ADV — см. adv_dollar_proxy_from_volatility.
    Возвращает (total, detail dict).
    """
    w = np.asarray(w, dtype=float).ravel()
    w_prev = np.asarray(w_prev, dtype=float).ravel()
    cov_mat = np.asarray(cov_mat, dtype=float)
    if len(w) != len(w_prev):
        raise ValueError('w and w_prev must have same length')
    dw = np.abs(w - w_prev) * float(notional)
    n = len(w)
    if adv_dollar is None:
        adv = adv_dollar_proxy_from_volatility(cov_mat)
    else:
        adv = np.asarray(adv_dollar, dtype=float).ravel()
        if len(adv) != n:
            raise ValueError('adv_dollar length must match w')
    part = dw / (adv + eps)
    part = np.minimum(part, float(participation_cap))
    temporary = float(eta_temp * np.sum(np.power(part, alpha)))
    permanent = float(eta_perm * np.sum(part))
    linear = float(linear_bps * 1e-4 * np.sum(dw))
    total = temporary + permanent + linear
    return total, {
        'temporary': temporary,
        'permanent': permanent,
        'linear_fee': linear,
        'participation': part,
        'adv_used': adv,
    }


def delay_opportunity_variance_proxy(w, w_prev, cov_mat):
    """Прокси риска задержки: (w-w_prev)' Sigma (w-w_prev)."""
    d = np.asarray(w, dtype=float).ravel() - np.asarray(w_prev, dtype=float).ravel()
    cov_mat = np.asarray(cov_mat, dtype=float)
    return float(d @ cov_mat @ d)


def total_is_penalty_for_optimizer(
    w,
    w_prev,
    cov_mat,
    adv_dollar=None,
    lambda_delay=0.0,
    **impact_kw,
):
    """
    Полная phi(w): AC-издержки + опционально lambda_delay * (w-w_prev)'Sigma(w-w_prev).
    """
    phi_imp, detail = expected_execution_impact_ac(
        w, w_prev, cov_mat, adv_dollar=adv_dollar, **impact_kw
    )
    delay_v = delay_opportunity_variance_proxy(w, w_prev, cov_mat)
    delay_term = float(lambda_delay) * delay_v
    total = phi_imp + delay_term
    detail = dict(detail)
    detail['delay_variance'] = delay_v
    detail['delay_penalty'] = delay_term
    detail['total_phi'] = total
    return total, detail


def neg_utility_mean_variance_is(
    w,
    mu,
    cov_mat,
    w_prev,
    lambda_risk,
    lambda_is,
    adv_dollar=None,
    lambda_delay=0.05,
    **impact_kw,
):
    """Минимизировать: -(mu'w - lambda_risk w'Sw - lambda_is * phi)."""
    phi, _ = total_is_penalty_for_optimizer(
        w, w_prev, cov_mat, adv_dollar=adv_dollar, lambda_delay=lambda_delay, **impact_kw
    )
    ret = portfolio_return(w, mu)
    risk = float(w @ np.asarray(cov_mat, dtype=float) @ w)
    return -(ret - float(lambda_risk) * risk - float(lambda_is) * phi)


def optimize_mean_variance_is(
    mu_vec,
    cov_mat,
    w_prev,
    lambda_risk=1.0,
    lambda_is=0.5,
    adv_dollar=None,
    lambda_delay=0.05,
    bounds=None,
    cons=None,
    x0=None,
    **impact_kw,
):
    """
    Максимизирует mu'w - lambda_risk w'Sw - lambda_is phi(w) при ограничениях (по умолчанию sum w = 1).
    """
    mu_vec = np.asarray(mu_vec, dtype=float).ravel()
    cov_mat = np.asarray(cov_mat, dtype=float)
    n = len(mu_vec)
    w_prev = np.asarray(w_prev, dtype=float).ravel()
    if len(w_prev) != n:
        raise ValueError('w_prev length must match mu_vec')
    if x0 is None:
        x0 = np.ones(n) / n
    if bounds is None:
        bounds = [(None, None)] * n
    if cons is None:
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

    def obj(w):
        return neg_utility_mean_variance_is(
            w,
            mu_vec,
            cov_mat,
            w_prev,
            lambda_risk,
            lambda_is,
            adv_dollar=adv_dollar,
            lambda_delay=lambda_delay,
            **impact_kw,
        )

    res = minimize(
        obj,
        x0,
        bounds=bounds,
        constraints=cons,
        method='SLSQP',
        options={'maxiter': 900, 'ftol': 1e-9},
    )
    return res


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


def get_moex_index_candles(ticker, start='2015-01-01', end='2025-12-31', session=None):
    """
    Дневные свечи индекса (IMOEX, MCFTR и т.д.) через ISS MOEX API.
    """
    if session is None:
        session = requests.Session()
        session.headers.update({'User-Agent': 'risk_utils/1.0'})
    url = f'https://iss.moex.com/iss/engines/stock/markets/index/securities/{ticker}/candles.json'
    parts = []
    start_offset = 0
    while True:
        params = {'from': start, 'till': end, 'interval': 24, 'start': start_offset}
        r = session.get(url, params=params, timeout=30)
        r.raise_for_status()
        j = r.json()
        data = j['candles']['data']
        cols = j['candles']['columns']
        if not data:
            break
        parts.append(pd.DataFrame(data, columns=cols))
        if len(data) < 100:
            break
        start_offset += len(data)
    if not parts:
        return pd.DataFrame()
    result = pd.concat(parts, ignore_index=True)
    result['begin'] = pd.to_datetime(result['begin'])
    result = result.set_index('begin')[['close']].rename(columns={'close': ticker})
    result.index.name = 'date'
    return result


def get_rebalance_dates(index, step='Y'):
    """
    Выбирает даты, на которых считаем mu и Sigma.
    'Y' / 'YE'  - конец года
    'Q' / 'QE'  - конец квартала
    'M' / 'ME'  - конец месяца
    'W'  - конец недели
    'D'  - каждый день
    Короткие алиасы ('Y','Q','M') приводятся к частотам pandas 2.x ('YE','QE','ME').
    """
    idx = pd.DatetimeIndex(index)
    if step == 'D':
        return idx
    aliases = {'Y': 'YE', 'Q': 'QE', 'M': 'ME'}
    if step in aliases:
        step = aliases[step]
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