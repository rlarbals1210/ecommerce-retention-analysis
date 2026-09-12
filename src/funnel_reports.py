"""Session event-presence summaries; no event ordering or product matching."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
import config

COUNTS = ["total_sessions", "cart_sessions", "purchase_sessions", "cart_and_purchase"]
TABLEAU_DIR = config.REPORTS_DIR / "tableau"


def summarize(monthly):
    """Recompute rates from integer counts, never averages of monthly rates."""
    df = monthly.copy()
    if "month" in df.columns:
        df = df.set_index("month")
    if df.empty or df.index.has_duplicates:
        raise ValueError("Expected one nonempty row per month")
    counts = df[COUNTS]
    if not np.isfinite(counts.to_numpy()).all() or (counts < 0).any().any() or (counts % 1 != 0).any().any():
        raise ValueError("Counts must be finite nonnegative integers")
    total, cart, purchase, both = (df[c] for c in COUNTS)
    if ((total <= 0) | (cart <= 0) | (cart > total) | (purchase > total) | (both > cart) | (both > purchase) | (cart + purchase - both > total)).any():
        raise ValueError("Invalid session sets or empty denominator")
    df = df[COUNTS].astype('int64')
    df['purchase_no_cart'] = purchase - both
    df['cart_share_of_sessions'] = cart / total
    df['purchase_share_within_cart'] = both / cart
    df['cart_without_purchase_rate'] = 1 - both / cart
    summed = df[COUNTS + ['purchase_no_cart']].sum()
    overall = pd.DataFrame({
        'step_order': [1, 2, 3],
        'stage': ['all_sessions', 'cart_present', 'cart_and_purchase'],
        'sessions': [summed.total_sessions, summed.cart_sessions, summed.cart_and_purchase],
        'pct_of_total_sessions': [1., summed.cart_sessions / summed.total_sessions, summed.cart_and_purchase / summed.total_sessions],
        'share_of_parent_set': [1., summed.cart_sessions / summed.total_sessions, summed.cart_and_purchase / summed.cart_sessions],
    })
    return df, overall


def style():
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ['AppleGothic', 'Noto Sans CJK KR', 'NanumGothic']:
        if name in available:
            plt.rcParams['font.family'] = name
            break
    plt.rcParams['axes.unicode_minus'] = False


def plot_overall(overall):
    style()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    labels = ['전체 세션', 'cart 있음', 'cart · purchase 모두 있음']
    shares = overall['pct_of_total_sessions'] * 100
    ax.barh(labels, shares, height=.52, color=['#cbd5e1', '#4f83cc', '#174a7e'])
    ax.invert_yaxis()
    for i, (pct, count) in enumerate(zip(shares, overall['sessions'])):
        ax.text(pct + 1.2, i, f'{pct:.1f}%  ({count:,}건)', va='center', fontsize=10)
    ax.set(xlim=(0, 133), xlabel='전체 세션 대비 비율 (%)')
    fig.text(.24, .94, '세션별 이벤트 존재 비율', fontsize=15)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.xaxis.grid(True, alpha=.15)
    ax.set_axisbelow(True)
    for s in ['top', 'right', 'left']:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis='y', length=0)
    fig.subplots_adjust(left=.24, right=.98, top=.83, bottom=.26)
    fig.text(.24, .87, '2019-10 ~ 2020-04 · 월별 고유 세션 수 합계', color='#475569', fontsize=10)
    fig.text(.03, .08, '동일 세션 내 이벤트 존재 기준 · 발생 순서·동일 상품·월 경계 중복 미검증', fontsize=9, color='#475569')
    fig.savefig(config.FIGURES_DIR / '03_funnel_overall.png', dpi=160)
    return fig


def plot_monthly(monthly):
    style()
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, label in [('cart_share_of_sessions', 'cart / 전체 세션'), ('purchase_share_within_cart', 'cart·purchase 모두 / cart'), ('cart_without_purchase_rate', 'cart 있고 purchase 없음 / cart')]:
        ax.plot(monthly.index, monthly[key], marker='o', label=label)
    ax.set(title='월별 세션 이벤트 존재 비율', xlabel='월', ylabel='비율')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
    ax.set_ylim(0, .75)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=.15)
    fig.subplots_adjust(bottom=.22)
    fig.text(.08, .07, '2019-10~11 cart 기록 이상 관찰 · 수집 변경 이력이 없어 원인 미확정', fontsize=9, color='#475569')
    fig.savefig(config.FIGURES_DIR / '03_funnel_monthly_trend.png', dpi=160)
    return fig


def export(monthly):
    monthly, overall = summarize(monthly)
    monthly.to_csv(TABLEAU_DIR / 'funnel_monthly.csv', encoding='utf-8-sig')
    overall.to_csv(TABLEAU_DIR / 'funnel_overall.csv', index=False, encoding='utf-8-sig')
    kpi = pd.read_csv(TABLEAU_DIR / 'kpi_summary.csv')
    kpi['metric'] = kpi['metric'].replace({
        '구매 전환율': '구매 경험자 비율', '확정 이탈률': '4월 미활동 비율',
        'view→cart 전환율': 'cart / 전체 세션', 'cart→purchase 전환율': 'cart·purchase 모두 / cart',
        '장바구니 이탈률': 'cart 미구매 비율',
    })
    totals = monthly[COUNTS].sum()
    values = {
        'cart / 전체 세션': (totals.cart_sessions / totals.total_sessions, 'cart 세션 / 월별 고유 세션 수 합계'),
        'cart·purchase 모두 / cart': (totals.cart_and_purchase / totals.cart_sessions, 'cart와 purchase가 모두 존재하는 세션 / cart 세션'),
        'cart 미구매 비율': (1 - totals.cart_and_purchase / totals.cart_sessions, 'cart가 있고 purchase가 없는 세션 / cart 세션; 순서·동일 상품 미검증'),
    }
    for metric, (value, note) in values.items():
        mask = kpi.metric.eq(metric)
        if mask.sum() != 1:
            raise ValueError(f'Expected exactly one KPI: {metric}')
        kpi.loc[mask, ['value', 'note']] = [value, note]
    kpi.loc[kpi.metric.eq('전체 세션'), 'note'] = '7개월 월별 고유 세션 수 합계; 월 경계 중복 미검증'
    kpi.loc[kpi.metric.eq('4월 미활동 비율'), 'note'] = '3/31 UTC까지 관측된 사용자 중 4월 미활동; 영구 이탈 아님; 4월 최초 관측자 제외'
    kpi.loc[kpi.metric.eq('판단 보류(censored)'), 'note'] = '4월 최초 관측자로 관찰 기회 부족; 실제 신규 가입자 아님'
    kpi.to_csv(TABLEAU_DIR / 'kpi_summary.csv', index=False, encoding='utf-8-sig')
    return monthly, overall


if __name__ == '__main__':
    monthly, overall = export(pd.read_csv(TABLEAU_DIR / 'funnel_monthly.csv'))
    plot_overall(overall)
    plot_monthly(monthly)
    print(overall.to_string(index=False))
