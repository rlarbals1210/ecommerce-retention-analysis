"""Candidate-backed Recency comparison for notebook 05; no raw CSV access."""
import json

import pandas as pd

import config
from src.ab_design import build_risk_population
from src.recency_reference import sha256_file
from src.recency_validation import daily_distribution, interval_counts


def load_comparison():
    report = json.loads(config.RECENCY_VALIDATION_PATH.read_text())
    if report.get('decision_status') != 'jointly_confirmed_branch_1':
        raise ValueError('Candidate requires confirmed joint review')
    paths = {'candidate': config.RECENCY_CANDIDATE_PATH,
             'labels': config.PROC_DIR / 'churn_labels.parquet',
             'buyers': config.PROC_DIR / 'purchase_users_at_reference.parquet'}
    for key, path in paths.items():
        if sha256_file(path) != report['input_files'][key]['sha256']:
            raise ValueError(f'Validated input changed: {key}')
    recency = pd.read_parquet(paths['candidate'], columns=['user_id', 'recency_at_ref'])
    labels = pd.read_parquet(paths['labels'], columns=['user_id', 'status'])
    labels['status'] = labels['status'].astype('category')
    buyers = pd.read_parquet(paths['buyers'], columns=['user_id'])['user_id']
    distribution = daily_distribution(recency, labels, buyers, 182)
    if distribution != report['daily_distributions']['candidate']:
        raise ValueError('Recomputed distribution differs from validated candidate')
    table = pd.DataFrame(interval_counts(distribution))
    risk = build_risk_population(recency, labels, buyers)
    return table, risk


def with_totals(table):
    """Totals are computed from numerators/denominators, never averaged rates."""
    totals = table.groupby('recency_band', sort=False)[['users', 'returned_users']].sum().reset_index()
    totals['baseline'] = totals['returned_users'].div(totals['users'].replace(0, float('nan')))
    totals['segment'] = 'overall'
    strata = table.copy()
    strata['segment'] = strata['is_buyer_at_reference'].map({True: 'buyer_before_reference', False: 'nonbuyer_before_reference'})
    columns = ['recency_band', 'segment', 'users', 'returned_users', 'baseline']
    return pd.concat([totals[columns], strata[columns]], ignore_index=True)


def comparison_markdown(table):
    names = {'overall': '전체', 'buyer_before_reference': '사전 구매', 'nonbuyer_before_reference': '사전 비구매'}
    lines = ['| Recency(일) | 구매층 | 대상자 수 | 4월 활동자 수 | 4월 활동률 |',
             '|---|---|---:|---:|---:|']
    for band in ['0-7', '8-14', '15-29', '30+']:
        for segment in names:
            r = table.loc[table.recency_band.eq(band) & table.segment.eq(segment)].iloc[0]
            label = band + (' (휴면 기준 충족군의 복귀)' if band == '30+' else '')
            rate = f'{r.baseline:.2%}' if pd.notna(r.baseline) else '해당 없음'
            lines.append(f'| {label} | {names[segment]} | {r.users:,} | {r.returned_users:,} | {rate} |')
    return '\n'.join(lines)


def check_risk_alignment(table, summary):
    a = table.loc[table.recency_band.eq('15-29')].set_index('segment')
    b = summary.set_index('segment')
    for column in ['users', 'returned_users', 'baseline']:
        if not a[column].sort_index().equals(b[column].sort_index()):
            raise ValueError(f'A/05 mismatch: {column}')
