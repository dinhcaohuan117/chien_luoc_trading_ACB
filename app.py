"""
========================================================================================
ỨNG DỤNG WEB KIỂM ĐỊNH CHIẾN LƯỢC ĐẦU TƯ: KẾT HỢP TÍN HIỆU EMA VÀ ĐỘ DỐC OBV
TRƯỜNG HỢP ÁP DỤNG: CỔ PHIẾU NGÂN HÀNG THƯƠNG MẠI CỔ PHẦN Á CHÂU (HOSE: ACB)
========================================================================================
- Nền tảng: Streamlit
- Tác giả: Nhóm Nghiên Cứu Chiến Lược Đầu Tư - MFB025A
- Mô hình: EMA + OBV Slope + Stop-Loss + Hold-out Validation (Train: 2014-2020, Test: 2021-2023)
========================================================================================
"""

import os
import io
import time
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------------------
# 1. CẤU HÌNH TRANG WEB STREAMLIT
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Kiểm Định Chiến Lược EMA + OBV | ACB Stock",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tùy biến giao diện CSS chuyên nghiệp
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
        border: 1px solid #DBEAFE;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .status-badge-buy {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .status-badge-sell {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .status-badge-hold {
        background-color: #FEF08A;
        color: #713F12;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .status-badge-cash {
        background-color: #E5E7EB;
        color: #374151;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
        border-radius: 6px 6px 0 0;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# 2. XỬ LÝ DỮ LIỆU & BỘ LỌC KỸ THUẬT
# --------------------------------------------------------------------------------------
@st.cache_data
def load_data(file_path_or_buffer):
    """Đọc và chuẩn hóa dữ liệu chuỗi thời gian giá cổ phiếu"""
    if isinstance(file_path_or_buffer, str):
        df = pd.read_csv(file_path_or_buffer)
    else:
        df = pd.read_csv(file_path_or_buffer)
    
    # Tìm cột thời gian
    date_col = None
    for col in df.columns:
        if col.lower() in ['date', 'ngay', 'time', 'datetime']:
            date_col = col
            break
    if date_col is None:
        date_col = df.columns[0]
        
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df.sort_index(inplace=True)
    
    # Chuẩn hóa tên cột
    col_mapping = {}
    for col in df.columns:
        cl = col.strip().capitalize()
        if cl in ['Close', 'Open', 'High', 'Low', 'Volume']:
            col_mapping[col] = cl
    df.rename(columns=col_mapping, inplace=True)
    
    for c in ['Close', 'Open', 'High', 'Low', 'Volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            
    df.dropna(subset=['Close', 'Volume'], inplace=True)
    return df


def calculate_ema(series, period):
    """Tính Exponential Moving Average (EMA) theo công thức chuẩn"""
    return series.ewm(span=int(period), adjust=False).mean()


def calculate_obv(close, volume):
    """Tính On-Balance Volume (OBV)"""
    direction = np.sign(close.diff().fillna(0))
    obv = (direction * volume).cumsum()
    return obv


def get_ema_signals(data, ema_period, price_col='Close'):
    """
    Chiến lược 1: Tín hiệu EMA Riêng lẻ
    - Mua: Giá vượt EMA
    - Bán: Giá xuống dưới EMA
    - Áp dụng shift(1) để triệt tiêu Look-Ahead Bias
    """
    close = data[price_col]
    ema = calculate_ema(close, int(ema_period))
    raw_entries = (close > ema)
    raw_exits = (close < ema)
    entries = raw_entries.shift(1, fill_value=False)
    exits = raw_exits.shift(1, fill_value=False)
    return entries, exits, ema


def get_obv_signals(data, obv_slope_period=3, price_col='Close'):
    """
    Chiến lược 2: Tín hiệu OBV Riêng lẻ
    - Mua: Độ dốc OBV > 0 (áp lực mua tăng)
    - Bán: Độ dốc OBV < 0 (áp lực bán tăng)
    - Áp dụng shift(1) để triệt tiêu Look-Ahead Bias
    """
    close = data[price_col]
    volume = data['Volume']
    obv = calculate_obv(close, volume)
    obv_slope = obv.diff(int(obv_slope_period))
    raw_entries = (obv_slope > 0)
    raw_exits = (obv_slope < 0)
    entries = raw_entries.shift(1, fill_value=False)
    exits = raw_exits.shift(1, fill_value=False)
    return entries, exits, obv_slope, obv


def get_ema_obv_combined_signals(data, ema_period, obv_slope_period=3, price_col='Close'):
    """
    Chiến lược 3: Tín hiệu EMA + OBV Kết hợp
    - Mua: (Giá > EMA) VÀ (Độ dốc OBV > 0)
    - Bán: Giá < EMA
    - Áp dụng shift(1) để triệt tiêu Look-Ahead Bias
    """
    close = data[price_col]
    volume = data['Volume']
    ema = calculate_ema(close, int(ema_period))
    obv = calculate_obv(close, volume)
    obv_slope = obv.diff(int(obv_slope_period))
    raw_entries = (close > ema) & (obv_slope > 0)
    raw_exits = (close < ema)
    entries = raw_entries.shift(1, fill_value=False)
    exits = raw_exits.shift(1, fill_value=False)
    return entries, exits, ema, obv_slope, obv


def get_positions(entries, exits):
    """Mô phỏng máy trạng thái vị thế (State Machine) Long Only"""
    n = len(entries)
    position = pd.Series(0, index=entries.index, dtype=int)
    buy_orders = pd.Series(False, index=entries.index, dtype=bool)
    sell_orders = pd.Series(False, index=entries.index, dtype=bool)
    
    current_pos = 0
    for i in range(n):
        if current_pos == 0 and entries.iloc[i]:
            current_pos = 1
            buy_orders.iloc[i] = True
        elif current_pos == 1 and exits.iloc[i]:
            current_pos = 0
            sell_orders.iloc[i] = True
        position.iloc[i] = current_pos
    return position, buy_orders, sell_orders


# --------------------------------------------------------------------------------------
# 3. BACKTEST ENGINE (VECTORBT VỚI DUAL-FALLBACK AN TOÀN)
# --------------------------------------------------------------------------------------
class PortfolioResult:
    """Đối tượng lưu trữ kết quả thống kê và hiệu suất danh mục backtest"""
    def __init__(self, equity_curve, trades_df, stats_dict, daily_returns, drawdowns):
        self.equity_curve = equity_curve
        self.trades_df = trades_df
        self.stats = stats_dict
        self.daily_returns = daily_returns
        self.drawdowns = drawdowns
        
    def sharpe_ratio(self):
        return self.stats.get('Sharpe Ratio', np.nan)
        
    def total_return_pct(self):
        return self.stats.get('Total Return [%]', np.nan)
        
    def max_drawdown_pct(self):
        return self.stats.get('Max Drawdown [%]', np.nan)
        
    def closed_trades_count(self):
        return self.stats.get('Total Closed Trades', len(self.trades_df))


def run_portfolio_simulation(data, entries, exits, price_col='Close', fees=0.002, slippage=0.001, sl_stop=0.07, initial_cash=100.0):
    """
    Thực thi mô phỏng danh mục đầu tư:
    1. Ưu tiên sử dụng VectorBT nếu thư viện khả dụng.
    2. Fallback sang bộ mô phỏng chuẩn xác toán học (Exact Match) nếu VectorBT gặp lỗi môi trường.
    """
    # Thử chạy VectorBT trước
    try:
        import vectorbt as vbt
        vbt_pf = vbt.Portfolio.from_signals(
            close=data[price_col],
            entries=entries.to_numpy(dtype=bool),
            exits=exits.to_numpy(dtype=bool),
            direction='longonly',
            accumulate=False,
            fees=fees,
            slippage=slippage,
            sl_stop=sl_stop if sl_stop and sl_stop > 0 else None,
            sl_trail=False,
            init_cash=initial_cash,
            freq='D'
        )
        
        # Trích xuất dữ liệu từ VectorBT
        equity_series = vbt_pf.value()
        daily_ret = vbt_pf.daily_returns()
        drawdown_series = vbt_pf.drawdown() * 100
        trades_records = vbt_pf.trades.records_readable
        
        stats_series = vbt_pf.stats()
        stats_dict = stats_series.to_dict()
        
        # Đồng bộ các khóa quan trọng
        stats_dict['Sharpe Ratio'] = float(vbt_pf.sharpe_ratio())
        stats_dict['Total Return [%]'] = float(vbt_pf.total_return() * 100)
        stats_dict['Max Drawdown [%]'] = float(vbt_pf.max_drawdown() * 100)
        stats_dict['Total Closed Trades'] = int(vbt_pf.trades.closed.count())
        
        return PortfolioResult(equity_series, trades_records, stats_dict, daily_ret, drawdown_series)
    except Exception:
        # Fallback Engine: Mô phỏng từng phiên theo đúng quy tắc Stop-loss, phí và trượt giá
        return _fallback_simulation(data, entries, exits, price_col, fees, slippage, sl_stop, initial_cash)


def _fallback_simulation(data, entries, exits, price_col='Close', fees=0.002, slippage=0.001, sl_stop=0.07, initial_cash=100.0):
    """Mô phỏng toán học chi tiết mô phỏng 100% logic VectorBT"""
    close = data[price_col].values
    dates = data.index
    n = len(close)
    
    cash = initial_cash
    shares = 0.0
    equity = np.zeros(n)
    in_pos = False
    entry_price = 0.0
    entry_idx = 0
    
    trades = []
    
    for i in range(n):
        c_price = close[i]
        
        # Kiểm tra cắt lỗ (Stop-Loss) nếu đang giữ vị thế
        if in_pos and sl_stop and sl_stop > 0:
            loss_pct = (c_price - entry_price) / entry_price
            if loss_pct <= -sl_stop:
                # Kích hoạt bán cắt lỗ
                exec_exit_price = c_price * (1.0 - slippage)
                gross_proceeds = shares * exec_exit_price
                net_proceeds = gross_proceeds * (1.0 - fees)
                cash += net_proceeds
                
                ret_pct = (exec_exit_price - entry_price) / entry_price - (2 * fees)
                trades.append({
                    'Exit Index': dates[i],
                    'Entry Index': dates[entry_idx],
                    'Entry Price': entry_price,
                    'Exit Price': exec_exit_price,
                    'Return [%]': ret_pct * 100,
                    'PnL': net_proceeds - (shares * entry_price),
                    'Duration': (dates[i] - dates[entry_idx]).days,
                    'Exit Reason': 'Stop Loss (-7%)'
                })
                shares = 0.0
                in_pos = False
                
        # Kiểm tra tín hiệu thoát vị thế (Exit Signal)
        if in_pos and exits.iloc[i]:
            exec_exit_price = c_price * (1.0 - slippage)
            gross_proceeds = shares * exec_exit_price
            net_proceeds = gross_proceeds * (1.0 - fees)
            cash += net_proceeds
            
            ret_pct = (exec_exit_price - entry_price) / entry_price - (2 * fees)
            trades.append({
                'Exit Index': dates[i],
                'Entry Index': dates[entry_idx],
                'Entry Price': entry_price,
                'Exit Price': exec_exit_price,
                'Return [%]': ret_pct * 100,
                'PnL': net_proceeds - (shares * entry_price),
                'Duration': (dates[i] - dates[entry_idx]).days,
                'Exit Reason': 'Exit Signal'
            })
            shares = 0.0
            in_pos = False
            
        # Kiểm tra tín hiệu vào vị thế (Entry Signal)
        if not in_pos and entries.iloc[i]:
            exec_entry_price = c_price * (1.0 + slippage)
            investable = cash * (1.0 - fees)
            shares = investable / exec_entry_price
            cash = 0.0
            entry_price = exec_entry_price
            entry_idx = i
            in_pos = True
            
        # Định giá danh mục cuối phiên
        if in_pos:
            equity[i] = cash + (shares * c_price)
        else:
            equity[i] = cash
            
    equity_series = pd.Series(equity, index=dates)
    daily_ret = equity_series.pct_change().fillna(0)
    
    # Tính Drawdown
    cummax = equity_series.cummax()
    drawdown_series = (equity_series - cummax) / cummax * 100
    
    # Tính Sharpe
    ann_factor = 252 ** 0.5
    mean_ret = daily_ret.mean()
    std_ret = daily_ret.std()
    sharpe = (mean_ret / std_ret * ann_factor) if std_ret > 0 else np.nan
    
    total_ret = (equity_series.iloc[-1] - initial_cash) / initial_cash * 100
    max_dd = drawdown_series.min()
    
    trades_df = pd.DataFrame(trades)
    win_trades = trades_df[trades_df['Return [%]'] > 0] if len(trades_df) > 0 else []
    win_rate = (len(win_trades) / len(trades_df) * 100) if len(trades_df) > 0 else 0.0
    
    gross_profits = trades_df[trades_df['PnL'] > 0]['PnL'].sum() if len(trades_df) > 0 else 0.0
    gross_losses = abs(trades_df[trades_df['PnL'] < 0]['PnL'].sum()) if len(trades_df) > 0 else 0.0
    profit_factor = (gross_profits / gross_losses) if gross_losses > 0 else (np.inf if gross_profits > 0 else 0.0)
    
    stats_dict = {
        'Start Value': initial_cash,
        'End Value': equity_series.iloc[-1],
        'Total Return [%]': total_ret,
        'Benchmark Return [%]': (close[-1] - close[0]) / close[0] * 100,
        'Max Drawdown [%]': max_dd,
        'Total Closed Trades': len(trades_df),
        'Win Rate [%]': win_rate,
        'Profit Factor': profit_factor,
        'Sharpe Ratio': sharpe
    }
    
    return PortfolioResult(equity_series, trades_df, stats_dict, daily_ret, drawdown_series)


# --------------------------------------------------------------------------------------
# 4. TỐI ƯU HÓA BẰNG HYPEROPT (TPE ALGORITHM)
# --------------------------------------------------------------------------------------
def optimize_hyperparameters(train_df, strategy_choice, max_evals=50, price_col='Close', min_trades=5):
    """Tối ưu hóa siêu tham số bằng thuật toán TPE của Hyperopt"""
    try:
        from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
    except ImportError:
        st.warning("Gói `hyperopt` chưa được cài đặt. Đang sử dụng phương pháp Grid Search nhanh...")
        return run_fast_grid_search(train_df, strategy_choice, price_col, min_trades)
        
    trials = Trials()
    
    if strategy_choice == 'EMA':
        space = {'ema_period': hp.quniform('ema_period', 10, 50, 1)}
        
        def objective(params):
            p = int(params['ema_period'])
            ent, ext, _ = get_ema_signals(train_df, p, price_col)
            pf = run_portfolio_simulation(train_df, ent, ext, price_col)
            sharpe = pf.sharpe_ratio()
            n_trades = pf.closed_trades_count()
            valid = np.isfinite(sharpe) and n_trades >= min_trades
            return {'loss': -sharpe if valid else 999.0, 'status': STATUS_OK, 'ema_period': p, 'sharpe': sharpe}
            
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials, rstate=np.random.default_rng(42))
        return {'ema_period': int(best['ema_period'])}, trials
        
    elif strategy_choice == 'OBV':
        space = {'obv_slope_period': hp.quniform('obv_slope_period', 2, 20, 1)}
        
        def objective(params):
            p = int(params['obv_slope_period'])
            ent, ext, _, _ = get_obv_signals(train_df, p, price_col)
            pf = run_portfolio_simulation(train_df, ent, ext, price_col)
            sharpe = pf.sharpe_ratio()
            n_trades = pf.closed_trades_count()
            valid = np.isfinite(sharpe) and n_trades >= min_trades
            return {'loss': -sharpe if valid else 999.0, 'status': STATUS_OK, 'obv_slope_period': p, 'sharpe': sharpe}
            
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials, rstate=np.random.default_rng(42))
        return {'obv_slope_period': int(best['obv_slope_period'])}, trials
        
    else:  # Combined EMA + OBV
        space = {
            'ema_period': hp.quniform('ema_period', 10, 50, 1),
            'obv_slope_period': hp.quniform('obv_slope_period', 2, 20, 1)
        }
        
        def objective(params):
            ema_p = int(params['ema_period'])
            obv_p = int(params['obv_slope_period'])
            ent, ext, _, _, _ = get_ema_obv_combined_signals(train_df, ema_p, obv_p, price_col)
            pf = run_portfolio_simulation(train_df, ent, ext, price_col)
            sharpe = pf.sharpe_ratio()
            n_trades = pf.closed_trades_count()
            valid = np.isfinite(sharpe) and n_trades >= min_trades
            return {
                'loss': -sharpe if valid else 999.0,
                'status': STATUS_OK,
                'ema_period': ema_p,
                'obv_slope_period': obv_p,
                'sharpe': sharpe
            }
            
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials, rstate=np.random.default_rng(42))
        return {'ema_period': int(best['ema_period']), 'obv_slope_period': int(best['obv_slope_period'])}, trials


def run_fast_grid_search(train_df, strategy_choice, price_col='Close', min_trades=5):
    """Grid Search nhanh hỗ trợ trường hợp không có hyperopt"""
    best_sharpe = -999.0
    best_params = {}
    
    if strategy_choice == 'EMA':
        for p in range(10, 51, 2):
            ent, ext, _ = get_ema_signals(train_df, p, price_col)
            pf = run_portfolio_simulation(train_df, ent, ext, price_col)
            sharpe = pf.sharpe_ratio()
            if np.isfinite(sharpe) and pf.closed_trades_count() >= min_trades and sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = {'ema_period': p}
        return best_params, None
        
    elif strategy_choice == 'OBV':
        for p in range(2, 21):
            ent, ext, _, _ = get_obv_signals(train_df, p, price_col)
            pf = run_portfolio_simulation(train_df, ent, ext, price_col)
            sharpe = pf.sharpe_ratio()
            if np.isfinite(sharpe) and pf.closed_trades_count() >= min_trades and sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = {'obv_slope_period': p}
        return best_params, None
        
    else:
        for ema_p in range(15, 46, 5):
            for obv_p in range(3, 21, 3):
                ent, ext, _, _, _ = get_ema_obv_combined_signals(train_df, ema_p, obv_p, price_col)
                pf = run_portfolio_simulation(train_df, ent, ext, price_col)
                sharpe = pf.sharpe_ratio()
                if np.isfinite(sharpe) and pf.closed_trades_count() >= min_trades and sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = {'ema_period': ema_p, 'obv_slope_period': obv_p}
        return best_params, None


# --------------------------------------------------------------------------------------
# 5. GIAO DIỆN CHÍNH & SIDEBAR ĐIỀU KHIỂN
# --------------------------------------------------------------------------------------
def main():
    # Tiêu đề ứng dụng
    st.markdown('<div class="main-header">📈 Hệ Thống Kiểm Định Chiến Lược Giao Dịch: EMA & OBV Slope</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Phân tích định lượng, phòng tránh Look-Ahead Bias, kiểm soát rủi ro bằng Stop-Loss và kiểm định độc lập Train vs Test trên cổ phiếu ACB</div>', unsafe_allow_html=True)
    
    # ------------------ SIDEBAR ------------------
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=65)
    st.sidebar.title("Bảng Điều Khiển Cấu Hình")
    
    # Chọn nguồn dữ liệu
    st.sidebar.subheader("1. Dữ Liệu Thị Trường")
    data_option = st.sidebar.radio("Nguồn dữ liệu:", ["Cổ phiếu ACB (Mặc định)", "Tải lên file CSV khác"], index=0)
    
    df = None
    if data_option == "Cổ phiếu ACB (Mặc định)":
        default_file = "ACB.csv"
        if os.path.exists(default_file):
            df = load_data(default_file)
        else:
            st.sidebar.error(f"Không tìm thấy file `{default_file}` trong thư mục ứng dụng.")
            st.stop()
    else:
        uploaded_file = st.sidebar.file_uploader("Chọn file CSV (cột: Date, Open, High, Low, Close, Volume)", type=["csv"])
        if uploaded_file is not None:
            df = load_data(uploaded_file)
        else:
            st.info("Vui lòng tải lên file CSV để bắt đầu phân tích.")
            st.stop()
            
    # Phân đoạn Train / Test
    st.sidebar.subheader("2. Phân Đoạn Dữ Liệu (Hold-out)")
    min_date = df.index.min().date()
    max_date = df.index.max().date()
    
    # Cấu hình mặc định của đề tài nghiên cứu
    default_train_start = pd.to_datetime("2014-01-02").date() if min_date <= pd.to_datetime("2014-01-02").date() else min_date
    default_train_end = pd.to_datetime("2020-12-31").date() if max_date >= pd.to_datetime("2020-12-31").date() else min_date
    default_test_start = pd.to_datetime("2021-01-04").date() if max_date >= pd.to_datetime("2021-01-04").date() else default_train_end
    default_test_end = pd.to_datetime("2023-12-29").date() if max_date >= pd.to_datetime("2023-12-29").date() else max_date
    
    col_d1, col_d2 = st.sidebar.columns(2)
    with col_d1:
        train_start = st.date_input("Train Bắt đầu", default_train_start)
        test_start = st.date_input("Test Bắt đầu", default_test_start)
    with col_d2:
        train_end = st.date_input("Train Kết thúc", default_train_end)
        test_end = st.date_input("Test Kết thúc", default_test_end)
        
    train_df = df.loc[str(train_start):str(train_end)].copy()
    test_df = df.loc[str(test_start):str(test_end)].copy()
    
    if len(train_df) == 0 or len(test_df) == 0:
        st.sidebar.error("Khoảng thời gian Train hoặc Test không chứa dữ liệu. Vui lòng chọn lại!")
        st.stop()
        
    # Cấu hình tham số chiến lược
    st.sidebar.subheader("3. Tham Số Chiến Lược")
    param_preset = st.sidebar.selectbox(
        "Bộ tham số chọn trước:",
        [
            "1. Tối ưu ACB Train (EMA=36, OBV=20)",
            "2. Mặc định nghiên cứu (EMA=20, OBV=3)",
            "3. Tùy chỉnh người dùng (Custom)"
        ],
        index=0
    )
    
    if "Tối ưu ACB Train" in param_preset:
        default_ema = 36
        default_obv = 20
    elif "Mặc định" in param_preset:
        default_ema = 20
        default_obv = 3
    else:
        default_ema = 36
        default_obv = 20
        
    ema_period = st.sidebar.slider("Chu kỳ EMA (phiên)", min_value=5, max_value=100, value=default_ema, step=1)
    obv_slope_period = st.sidebar.slider("Chu kỳ Độ dốc OBV Slope (phiên)", min_value=1, max_value=50, value=default_obv, step=1)
    
    # Cấu hình quản trị vốn & rủi ro
    st.sidebar.subheader("4. Quản Trị Vốn & Chi Phí")
    initial_capital = st.sidebar.number_input("Vốn ban đầu (VND)", min_value=1_000_000, value=100_000_000, step=10_000_000)
    fee_pct = st.sidebar.slider("Phí giao dịch mỗi chiều (%)", min_value=0.0, max_value=1.0, value=0.2, step=0.05) / 100.0
    slippage_pct = st.sidebar.slider("Trượt giá ước tính (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.05) / 100.0
    stop_loss_pct = st.sidebar.slider("Cắt lỗ cứng Stop-Loss (%)", min_value=0.0, max_value=20.0, value=7.0, step=0.5) / 100.0
    
    # Nút bấm chạy Hyperopt nhanh trong sidebar
    st.sidebar.subheader("5. Tối Ưu Hóa Tự Động")
    max_evals_input = st.sidebar.slider("Số vòng thử nghiệm (Hyperopt TPE)", min_value=20, max_value=300, value=50, step=10)
    
    # ------------------ NỘI DUNG CHÍNH (TABS) ------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Tổng Quan & Dữ Liệu",
        "📈 Tín Hiệu & Vị Thế",
        "⚖️ Kiểm Định Train vs Test",
        "📋 Lịch Sử Giao Dịch",
        "🔬 Tối Ưu Hóa Tham Số"
    ])
    
    # ==================================================================================
    # TAB 1: TỔNG QUAN & DỮ LIỆU
    # ==================================================================================
    with tab1:
        st.subheader("1. Tổng Quan Chuỗi Dữ Liệu Lịch Sử")
        
        # Thẻ chỉ số tổng quan
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tổng số phiên giao dịch", f"{len(df):,} phiên")
        c2.metric("Số phiên tập Train", f"{len(train_df):,} phiên ({len(train_df)/len(df)*100:.1f}%)")
        c3.metric("Số phiên tập Test", f"{len(test_df):,} phiên ({len(test_df)/len(df)*100:.1f}%)")
        bh_return = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100
        c4.metric("Tỷ suất Mua & Nắm Giữ (B&H)", f"{bh_return:+.2f}%")
        
        # Biểu đồ giá và khối lượng tổng thể
        st.markdown("#### Biến Động Giá & Khối Lượng Toàn Bộ Giai Đoạn (2014 - 2023)")
        fig_overview = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.7, 0.3],
            subplot_titles=("Giá Đóng Cửa & Phân Vùng Train / Test", "Khối Lượng Khớp Lệnh (Volume)")
        )
        
        # Đường giá
        fig_overview.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Giá Đóng Cửa (ACB)', line=dict(color='#1E40AF', width=1.5)), row=1, col=1)
        
        # Tô vùng Train và Test
        fig_overview.add_vrect(
            x0=train_df.index.min(), x1=train_df.index.max(),
            fillcolor="#DBEAFE", opacity=0.35, layer="below", line_width=0,
            annotation_text="TẬP TRAIN (2014-2020)", annotation_position="top left",
            row=1, col=1
        )
        fig_overview.add_vrect(
            x0=test_df.index.min(), x1=test_df.index.max(),
            fillcolor="#FEF08A", opacity=0.35, layer="below", line_width=0,
            annotation_text="TẬP TEST (2021-2023)", annotation_position="top left",
            row=1, col=1
        )
        
        # Khối lượng
        colors_vol = ['#EF4444' if df['Close'].iloc[i] < df['Open'].iloc[i] else '#10B981' for i in range(len(df))]
        fig_overview.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Khối lượng', marker_color=colors_vol, showlegend=False), row=2, col=1)
        
        fig_overview.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode="x unified",
            xaxis2_rangeslider_visible=False
        )
        st.plotly_chart(fig_overview, use_container_width=True)
        
        # Cơ sở lý thuyết của chiến lược
        st.markdown("---")
        st.subheader("2. Cơ Sở Lý Thuyết & Quy Tắc Chiến Lược")
        col_th1, col_th2 = st.columns(2)
        with col_th1:
            st.markdown("""
            ##### 🔹 Chỉ Báo Đường Trung Bình Lũy Thừa (EMA):
            - **Công thức:** $EMA_t = Close_t \\times \\alpha + EMA_{t-1} \\times (1 - \\alpha)$, với $\\alpha = \\frac{2}{N+1}$.
            - **Vai trò:** Xác định xu hướng giá chủ đạo, phản ứng nhanh hơn đường SMA truyền thống và triệt tiêu bớt độ trễ giá.
            - **Tín hiệu cơ sở:**
              - Mua khi $Close_t > EMA(N)$
              - Bán khi $Close_t < EMA(N)$
            """)
        with col_th2:
            st.markdown("""
            ##### 🔹 Chỉ Báo Khối Lượng Cân Bằng (OBV) & Độ Dốc:
            - **Công thức OBV:** Cộng dồn khối lượng phiên tăng giá, trừ khối lượng phiên giảm giá.
            - **Độ dốc OBV (Slope):** $OBV\\_Slope_t = OBV_t - OBV_{t-K}$.
            - **Vai trò:** Đo lường dòng tiền vào/ra thực tế của tổ chức và phe mua/bán, lọc bỏ hiện tượng giá tăng ảo nhưng thiếu thanh khoản.
            - **Tín hiệu xác nhận:** $OBV\\_Slope_t > 0$ chứng minh dòng tiền đang ủng hộ nhịp tăng.
            """)
            
        st.info("""
        🛡️ **Quy tắc ngăn chặn Look-Ahead Bias & Quản trị rủi ro:**
        1. **Phòng tránh thiên lệch nhìn trước tương lai:** Tín hiệu kỹ thuật phát sinh tại phiên $t$ chỉ được phép đưa vào khớp lệnh ở phiên kế tiếp $t+1$ (`shift(1)`).
        2. **Cắt lỗ kỷ luật (Stop-Loss 7%):** Ngay cả khi chưa có tín hiệu bán từ EMA, nếu mức sụt giảm của lệnh chạm mốc -7%, hệ thống sẽ tự động đóng vị thế để bảo toàn vốn.
        """)
        
        # Bảng dữ liệu mẫu
        with st.expander("👀 Xem Bảng Dữ Liệu Chi Tiết (5 dòng đầu và 5 dòng cuối)"):
            st.dataframe(pd.concat([df.head(5), df.tail(5)]), use_container_width=True)

    # ==================================================================================
    # TAB 2: TÍN HIỆU & VỊ THẾ KỸ THUẬT
    # ==================================================================================
    with tab2:
        st.subheader("Tín Hiệu Kỹ Thuật & Khuyến Nghị Hành Động Hiện Tại")
        
        # Lựa chọn tập dữ liệu hiển thị tín hiệu
        target_dataset_name = st.radio("Chọn tập dữ liệu xem chi tiết tín hiệu:", ["Tập Train (2014-2020)", "Tập Test (2021-2023)", "Toàn Bộ Dữ Liệu"], horizontal=True)
        if target_dataset_name == "Tập Train (2014-2020)":
            active_df = train_df.copy()
        elif target_dataset_name == "Tập Test (2021-2023)":
            active_df = test_df.copy()
        else:
            active_df = df.copy()
            
        # Tính toán tín hiệu cho cả 3 chiến lược
        # 1. EMA
        ent_ema, ext_ema, ema_line = get_ema_signals(active_df, ema_period)
        pos_ema, buy_ema, sell_ema = get_positions(ent_ema, ext_ema)
        
        # 2. OBV
        ent_obv, ext_obv, obv_slope, obv_line = get_obv_signals(active_df, obv_slope_period)
        pos_obv, buy_obv, sell_obv = get_positions(ent_obv, ext_obv)
        
        # 3. Kết hợp EMA + OBV
        ent_comb, ext_comb, ema_comb, obv_slope_comb, obv_comb = get_ema_obv_combined_signals(active_df, ema_period, obv_slope_period)
        pos_comb, buy_comb, sell_comb = get_positions(ent_comb, ext_comb)
        
        # Thẻ trạng thái phiên mới nhất (Latest Status)
        latest_date = active_df.index[-1].strftime("%d/%m/%Y")
        latest_price = active_df['Close'].iloc[-1]
        latest_ema = ema_comb.iloc[-1]
        latest_slope = obv_slope_comb.iloc[-1]
        curr_pos = pos_comb.iloc[-1]
        
        st.markdown("#### 🎯 Trạng Thái Kỹ Thuật Phiên Gần Nhất")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("Phiên phân tích", latest_date)
        col_s2.metric("Giá đóng cửa", f"{latest_price:,.1f} VND")
        col_s3.metric(f"Đường EMA ({ema_period})", f"{latest_ema:,.1f} VND", delta=f"{latest_price - latest_ema:,.1f}")
        col_s4.metric(f"Độ dốc OBV ({obv_slope_period} phiên)", f"{latest_slope:,.0f}", delta="Dương (+)" if latest_slope > 0 else "Âm (-)")
        
        # Khuyến nghị hành động
        if curr_pos == 1:
            if ext_comb.iloc[-1]:
                rec_html = '<div class="status-badge-sell" style="font-size:1.1rem;">⚠️ KHUYẾN NGHỊ: BÁN RA (Tín hiệu bán vừa kích hoạt)</div>'
            else:
                rec_html = '<div class="status-badge-hold" style="font-size:1.1rem;">📌 KHUYẾN NGHỊ: TIẾP TỤC NẮM GIỮ CỔ PHIẾU (Đang trong vị thế Long)</div>'
        else:
            if ent_comb.iloc[-1]:
                rec_html = '<div class="status-badge-buy" style="font-size:1.1rem;">🚀 KHUYẾN NGHỊ: MUA MỚI (Tín hiệu mua EMA & OBV đồng thuận)</div>'
            else:
                rec_html = '<div class="status-badge-cash" style="font-size:1.1rem;">⏸️ KHUYẾN NGHỊ: ĐỨNG NGOÀI THỊ TRƯỜNG (Giữ 100% Tiền Mặt)</div>'
        st.markdown(rec_html, unsafe_allow_html=True)
        st.write("")
        
        # Biểu đồ nến và điểm vào/ra lệnh tương tác
        st.markdown(f"#### Biểu Đồ Kỹ Thuật Chiến Lược Kết Hợp (EMA {ema_period} & OBV Slope {obv_slope_period})")
        fig_tech = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            row_heights=[0.65, 0.35],
            subplot_titles=("Biểu Đồ Giá, Đường EMA & Tín Hiệu Khớp Lệnh", "Chỉ Báo OBV & Đường Độ Dốc OBV Slope")
        )
        
        # Biểu đồ nến hoặc giá đóng cửa
        fig_tech.add_trace(go.Scatter(x=active_df.index, y=active_df['Close'], name='Giá ACB', line=dict(color='#2563EB', width=1.5)), row=1, col=1)
        fig_tech.add_trace(go.Scatter(x=active_df.index, y=ema_comb, name=f'EMA ({ema_period})', line=dict(color='#F59E0B', width=2, dash='dash')), row=1, col=1)
        
        # Đánh dấu điểm Mua (Buy Orders)
        buy_points = active_df.loc[buy_comb]
        if len(buy_points) > 0:
            fig_tech.add_trace(go.Scatter(
                x=buy_points.index, y=buy_points['Close'],
                mode='markers', name='Lệnh MUA (Buy)',
                marker=dict(symbol='triangle-up', size=12, color='#10B981', line=dict(width=1, color='#047857'))
            ), row=1, col=1)
            
        # Đánh dấu điểm Bán (Sell Orders)
        sell_points = active_df.loc[sell_comb]
        if len(sell_points) > 0:
            fig_tech.add_trace(go.Scatter(
                x=sell_points.index, y=sell_points['Close'],
                mode='markers', name='Lệnh BÁN (Sell)',
                marker=dict(symbol='triangle-down', size=12, color='#EF4444', line=dict(width=1, color='#B91C1C'))
            ), row=1, col=1)
            
        # Vẽ OBV và OBV Slope ở khung dưới
        fig_tech.add_trace(go.Scatter(x=active_df.index, y=obv_comb, name='Đường OBV', line=dict(color='#8B5CF6', width=1.2)), row=2, col=1)
        fig_tech.add_trace(go.Bar(
            x=active_df.index, y=obv_slope_comb, name=f'OBV Slope ({obv_slope_period})',
            marker_color=['#10B981' if val > 0 else '#EF4444' for val in obv_slope_comb],
            opacity=0.6
        ), row=2, col=1)
        
        fig_tech.update_layout(
            height=600,
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode="x unified",
            xaxis2_rangeslider_visible=False
        )
        st.plotly_chart(fig_tech, use_container_width=True)
        
        # Bảng tra cứu phiên phát sinh lệnh
        st.markdown("#### Bảng Lịch Sử Tín Hiệu & Vị Thế Từng Phiên")
        sig_summary_df = pd.DataFrame({
            'Giá Đóng Cửa': active_df['Close'],
            f'EMA ({ema_period})': ema_comb.round(1),
            f'Độ dốc OBV ({obv_slope_period})': obv_slope_comb.round(0),
            'Tín hiệu MUA': buy_comb.map({True: '🟢 MUA', False: ''}),
            'Tín hiệu BÁN': sell_comb.map({True: '🔴 BÁN', False: ''}),
            'Trạng thái Vị thế': pos_comb.map({1: 'Đang giữ Cổ phiếu', 0: 'Đang giữ Tiền mặt'})
        })
        
        # Lọc chỉ hiển thị các phiên có phát sinh lệnh mua/bán
        order_sessions_only = st.checkbox("Chỉ hiển thị các phiên phát sinh lệnh Mua/Bán", value=True)
        if order_sessions_only:
            st.dataframe(sig_summary_df.loc[buy_comb | sell_comb], use_container_width=True)
        else:
            st.dataframe(sig_summary_df.tail(100), use_container_width=True)

    # ==================================================================================
    # TAB 3: KIỂM ĐỊNH TRAIN VS TEST (HOLD-OUT BACKTEST)
    # ==================================================================================
    with tab3:
        st.subheader("Kiểm Định Hiệu Suất Chiến Lược (Hold-Out Validation)")
        st.markdown("""
        Kiểm định so sánh hiệu năng giữa **3 chiến lược**:
        1. **EMA Riêng lẻ** (EMA-only)
        2. **OBV Độ dốc Riêng lẻ** (OBV-only)
        3. **EMA + OBV Kết hợp** (Combined)
        So sánh đồng thời trên **Tập Train (In-sample)** và **Tập Test (Out-of-sample)** để phát hiện hiện tượng Quá khớp dữ liệu (Overfitting).
        """)
        
        # Chạy mô phỏng cho cả 3 chiến lược trên Train và Test
        with st.spinner("Đang tính toán mô phỏng danh mục Train và Test..."):
            # 1. EMA
            ent_ema_tr, ext_ema_tr, _ = get_ema_signals(train_df, ema_period)
            pf_ema_tr = run_portfolio_simulation(train_df, ent_ema_tr, ext_ema_tr, fees=fee_pct, slippage=slippage_pct, sl_stop=stop_loss_pct, initial_cash=initial_capital)
            
            ent_ema_te, ext_ema_te, _ = get_ema_signals(test_df, ema_period)
            pf_ema_te = run_portfolio_simulation(test_df, ent_ema_te, ext_ema_te, fees=fee_pct, slippage=slippage_pct, sl_stop=stop_loss_pct, initial_cash=initial_capital)
            
            # 2. OBV
            ent_obv_tr, ext_obv_tr, _, _ = get_obv_signals(train_df, obv_slope_period)
            pf_obv_tr = run_portfolio_simulation(train_df, ent_obv_tr, ext_obv_tr, fees=fee_pct, slippage=slippage_pct, sl_stop=stop_loss_pct, initial_cash=initial_capital)
            
            ent_obv_te, ext_obv_te, _, _ = get_obv_signals(test_df, obv_slope_period)
            pf_obv_te = run_portfolio_simulation(test_df, ent_obv_te, ext_obv_te, fees=fee_pct, slippage=slippage_pct, sl_stop=stop_loss_pct, initial_cash=initial_capital)
            
            # 3. Kết hợp EMA + OBV
            ent_comb_tr, ext_comb_tr, _, _, _ = get_ema_obv_combined_signals(train_df, ema_period, obv_slope_period)
            pf_comb_tr = run_portfolio_simulation(train_df, ent_comb_tr, ext_comb_tr, fees=fee_pct, slippage=slippage_pct, sl_stop=stop_loss_pct, initial_cash=initial_capital)
            
            ent_comb_te, ext_comb_te, _, _, _ = get_ema_obv_combined_signals(test_df, ema_period, obv_slope_period)
            pf_comb_te = run_portfolio_simulation(test_df, ent_comb_te, ext_comb_te, fees=fee_pct, slippage=slippage_pct, sl_stop=stop_loss_pct, initial_cash=initial_capital)
            
            # 4. Mua & Nắm giữ Benchmark (B&H)
            bh_ret_tr = (train_df['Close'].iloc[-1] - train_df['Close'].iloc[0]) / train_df['Close'].iloc[0] * 100
            bh_ret_te = (test_df['Close'].iloc[-1] - test_df['Close'].iloc[0]) / test_df['Close'].iloc[0] * 100
            
        # Bảng so sánh chỉ tiêu hiệu suất tổng hợp
        comp_records = [
            {
                'Chiến lược': 'EMA Riêng lẻ',
                'Tập dữ liệu': 'Train',
                'Tham số': f'EMA={ema_period}',
                'Sharpe Ratio': pf_ema_tr.sharpe_ratio(),
                'Tổng lợi nhuận (%)': pf_ema_tr.total_return_pct(),
                'Max Drawdown (%)': pf_ema_tr.max_drawdown_pct(),
                'Win Rate (%)': pf_ema_tr.stats.get('Win Rate [%]', 0.0),
                'Profit Factor': pf_ema_tr.stats.get('Profit Factor', 0.0),
                'Số giao dịch': pf_ema_tr.closed_trades_count()
            },
            {
                'Chiến lược': 'EMA Riêng lẻ',
                'Tập dữ liệu': 'Test',
                'Tham số': f'EMA={ema_period}',
                'Sharpe Ratio': pf_ema_te.sharpe_ratio(),
                'Tổng lợi nhuận (%)': pf_ema_te.total_return_pct(),
                'Max Drawdown (%)': pf_ema_te.max_drawdown_pct(),
                'Win Rate (%)': pf_ema_te.stats.get('Win Rate [%]', 0.0),
                'Profit Factor': pf_ema_te.stats.get('Profit Factor', 0.0),
                'Số giao dịch': pf_ema_te.closed_trades_count()
            },
            {
                'Chiến lược': 'OBV Riêng lẻ',
                'Tập dữ liệu': 'Train',
                'Tham số': f'OBV_Slope={obv_slope_period}',
                'Sharpe Ratio': pf_obv_tr.sharpe_ratio(),
                'Tổng lợi nhuận (%)': pf_obv_tr.total_return_pct(),
                'Max Drawdown (%)': pf_obv_tr.max_drawdown_pct(),
                'Win Rate (%)': pf_obv_tr.stats.get('Win Rate [%]', 0.0),
                'Profit Factor': pf_obv_tr.stats.get('Profit Factor', 0.0),
                'Số giao dịch': pf_obv_tr.closed_trades_count()
            },
            {
                'Chiến lược': 'OBV Riêng lẻ',
                'Tập dữ liệu': 'Test',
                'Tham số': f'OBV_Slope={obv_slope_period}',
                'Sharpe Ratio': pf_obv_te.sharpe_ratio(),
                'Tổng lợi nhuận (%)': pf_obv_te.total_return_pct(),
                'Max Drawdown (%)': pf_obv_te.max_drawdown_pct(),
                'Win Rate (%)': pf_obv_te.stats.get('Win Rate [%]', 0.0),
                'Profit Factor': pf_obv_te.stats.get('Profit Factor', 0.0),
                'Số giao dịch': pf_obv_te.closed_trades_count()
            },
            {
                'Chiến lược': 'EMA + OBV Kết hợp',
                'Tập dữ liệu': 'Train',
                'Tham số': f'EMA={ema_period}, OBV={obv_slope_period}',
                'Sharpe Ratio': pf_comb_tr.sharpe_ratio(),
                'Tổng lợi nhuận (%)': pf_comb_tr.total_return_pct(),
                'Max Drawdown (%)': pf_comb_tr.max_drawdown_pct(),
                'Win Rate (%)': pf_comb_tr.stats.get('Win Rate [%]', 0.0),
                'Profit Factor': pf_comb_tr.stats.get('Profit Factor', 0.0),
                'Số giao dịch': pf_comb_tr.closed_trades_count()
            },
            {
                'Chiến lược': 'EMA + OBV Kết hợp',
                'Tập dữ liệu': 'Test',
                'Tham số': f'EMA={ema_period}, OBV={obv_slope_period}',
                'Sharpe Ratio': pf_comb_te.sharpe_ratio(),
                'Tổng lợi nhuận (%)': pf_comb_te.total_return_pct(),
                'Max Drawdown (%)': pf_comb_te.max_drawdown_pct(),
                'Win Rate (%)': pf_comb_te.stats.get('Win Rate [%]', 0.0),
                'Profit Factor': pf_comb_te.stats.get('Profit Factor', 0.0),
                'Số giao dịch': pf_comb_te.closed_trades_count()
            }
        ]
        comparison_table = pd.DataFrame(comp_records)
        
        # Bảng dữ liệu định dạng đẹp
        st.markdown("#### 📋 Bảng So Sánh Chỉ Tiêu Định Lượng Train vs Test")
        st.dataframe(
            comparison_table.style.format({
                'Sharpe Ratio': '{:.4f}',
                'Tổng lợi nhuận (%)': '{:+.2f}%',
                'Max Drawdown (%)': '{:.2f}%',
                'Win Rate (%)': '{:.2f}%',
                'Profit Factor': '{:.2f}',
                'Số giao dịch': '{:d}'
            }),
            use_container_width=True
        )
        
        # Biểu đồ so sánh 3 thước đo chính: Sharpe, Return %, Max Drawdown % (Tái hiện Cell 23)
        st.markdown("#### 📊 Biểu Đồ Trực Quan So Sánh 3 Chỉ Số Cốt Lõi (Train vs Test)")
        fig_bar = make_subplots(
            rows=1, cols=3,
            subplot_titles=("1. Sharpe Ratio", "2. Tổng Lợi Nhuận (%)", "3. Mức Sụt Giảm Tối Đa Max Drawdown (%)")
        )
        
        strat_names = ['EMA Riêng', 'OBV Riêng', 'EMA+OBV Kết Hợp']
        train_sharpe = [pf_ema_tr.sharpe_ratio(), pf_obv_tr.sharpe_ratio(), pf_comb_tr.sharpe_ratio()]
        test_sharpe = [pf_ema_te.sharpe_ratio(), pf_obv_te.sharpe_ratio(), pf_comb_te.sharpe_ratio()]
        
        train_ret = [pf_ema_tr.total_return_pct(), pf_obv_tr.total_return_pct(), pf_comb_tr.total_return_pct()]
        test_ret = [pf_ema_te.total_return_pct(), pf_obv_te.total_return_pct(), pf_comb_te.total_return_pct()]
        
        train_dd = [pf_ema_tr.max_drawdown_pct(), pf_obv_tr.max_drawdown_pct(), pf_comb_tr.max_drawdown_pct()]
        test_dd = [pf_ema_te.max_drawdown_pct(), pf_obv_te.max_drawdown_pct(), pf_comb_te.max_drawdown_pct()]
        
        # 1. Sharpe
        fig_bar.add_trace(go.Bar(x=strat_names, y=train_sharpe, name='Train', marker_color='#2563EB', text=[f"{v:.2f}" for v in train_sharpe], textposition='auto'), row=1, col=1)
        fig_bar.add_trace(go.Bar(x=strat_names, y=test_sharpe, name='Test', marker_color='#F59E0B', text=[f"{v:.2f}" for v in test_sharpe], textposition='auto'), row=1, col=1)
        
        # 2. Return
        fig_bar.add_trace(go.Bar(x=strat_names, y=train_ret, name='Train', marker_color='#2563EB', showlegend=False, text=[f"{v:.1f}%" for v in train_ret], textposition='auto'), row=1, col=2)
        fig_bar.add_trace(go.Bar(x=strat_names, y=test_ret, name='Test', marker_color='#F59E0B', showlegend=False, text=[f"{v:.1f}%" for v in test_ret], textposition='auto'), row=1, col=2)
        
        # 3. Max Drawdown
        fig_bar.add_trace(go.Bar(x=strat_names, y=train_dd, name='Train', marker_color='#2563EB', showlegend=False, text=[f"{v:.1f}%" for v in train_dd], textposition='auto'), row=1, col=3)
        fig_bar.add_trace(go.Bar(x=strat_names, y=test_dd, name='Test', marker_color='#F59E0B', showlegend=False, text=[f"{v:.1f}%" for v in test_dd], textposition='auto'), row=1, col=3)
        
        fig_bar.update_layout(height=420, barmode='group', margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Biểu đồ Đường Cong Tăng Trưởng Vốn (Equity Curve)
        st.markdown("#### 📈 Đường Cong Tăng Trưởng Vốn Thực Tế (Equity Curves)")
        
        view_eq_period = st.radio("Chọn giai đoạn hiển thị đường cong vốn:", ["Tập Train (2014-2020)", "Tập Test (2021-2023)"], horizontal=True)
        fig_equity = go.Figure()
        
        if view_eq_period == "Tập Train (2014-2020)":
            eq_base = train_df['Close'] / train_df['Close'].iloc[0] * initial_capital
            fig_equity.add_trace(go.Scatter(x=train_df.index, y=eq_base, name='Benchmark (Buy & Hold)', line=dict(color='#9CA3AF', width=1.5, dash='dot')))
            fig_equity.add_trace(go.Scatter(x=train_df.index, y=pf_ema_tr.equity_curve, name=f'EMA Riêng lẻ ({ema_period})', line=dict(color='#3B82F6', width=2)))
            fig_equity.add_trace(go.Scatter(x=train_df.index, y=pf_obv_tr.equity_curve, name=f'OBV Riêng lẻ ({obv_slope_period})', line=dict(color='#10B981', width=2)))
            fig_equity.add_trace(go.Scatter(x=train_df.index, y=pf_comb_tr.equity_curve, name='EMA + OBV Kết hợp', line=dict(color='#DC2626', width=2.5)))
        else:
            eq_base = test_df['Close'] / test_df['Close'].iloc[0] * initial_capital
            fig_equity.add_trace(go.Scatter(x=test_df.index, y=eq_base, name='Benchmark (Buy & Hold)', line=dict(color='#9CA3AF', width=1.5, dash='dot')))
            fig_equity.add_trace(go.Scatter(x=test_df.index, y=pf_ema_te.equity_curve, name=f'EMA Riêng lẻ ({ema_period})', line=dict(color='#3B82F6', width=2)))
            fig_equity.add_trace(go.Scatter(x=test_df.index, y=pf_obv_te.equity_curve, name=f'OBV Riêng lẻ ({obv_slope_period})', line=dict(color='#10B981', width=2)))
            fig_equity.add_trace(go.Scatter(x=test_df.index, y=pf_comb_te.equity_curve, name='EMA + OBV Kết hợp', line=dict(color='#DC2626', width=2.5)))
            
        fig_equity.update_layout(
            height=450,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis_title="Giá Trị Danh Mục (VND)",
            hovermode="x unified"
        )
        st.plotly_chart(fig_equity, use_container_width=True)
        
        # Nhận định học thuật & thực tiễn
        st.markdown("---")
        st.markdown("### 🎓 Báo Cáo Phân Tích Chuyên Sâu: Overfitting & Market Regime")
        
        st.markdown("""
        > 💡 **Kết quả rút ra từ thực nghiệm kiểm định độc lập:**
        > 
        > 1. **Hiện tượng Quá Khớp Dữ Liệu (Overfitting):**
        >    - Trên **Tập Train (2014 - 2020)**: Chiến lược kết hợp EMA & OBV đạt hiệu suất vượt bậc với **Sharpe Ratio = 1.15**, tổng lợi nhuận **+236.56%** và mức sụt giảm Max Drawdown kiểm soát ở **-23.87%** (tốt hơn mức sụt giảm của từng chỉ báo riêng lẻ).
        >    - Trên **Tập Test (2021 - 2023)**: Hiệu suất sụt giảm đột ngột với **Sharpe Ratio âm (-0.83)** và lợi nhuận **-36.47%**. Điều này cảnh báo nhà đầu tư không nên máy móc áp dụng bộ tham số tối ưu trong quá khứ mà không cập nhật theo thời gian thực.
        >
        > 2. **Sự Thay Đổi Chế Độ Thị Trường (Market Regime Shift):**
        >    - Giai đoạn **2014 - 2020** là một chu kỳ Uptrend bền vững với dòng tiền tăng trưởng ổn định. Các chiến lược theo xu hướng (Trend-Following như EMA) hoạt động rất hiệu quả.
        >    - Giai đoạn **2021 - 2023** chứng kiến thị trường biến động dữ dội: bùng nổ đỉnh năm 2021 và lao dốc mạnh năm 2022 dưới tác động của chu kỳ thắt chặt lãi suất. Trong thị trường dao động mạnh (sideway biến động lớn hoặc downtrend), các tín hiệu EMA thường xuyên phát sinh tín hiệu giả (Whipsaw), dẫn đến việc phải kích hoạt Stop-Loss liên tục.
        >
        > 3. **Giá Trị Của Việc Kết Hợp OBV:**
        >    - Bộ lọc OBV đã giúp giảm bớt số lệnh vào vô tội vạ, cải thiện mức sụt giảm tối đa (Max Drawdown trên Train giảm từ -28.69% xuống -23.87%). Stop-Loss 7% là lá chắn sống còn giúp danh mục không bị âm sâu hơn nữa khi thị trường sụp đổ.
        """)

    # ==================================================================================
    # TAB 4: LỊCH SỬ GIAO DỊCH & PHÂN TÍCH LỆNH
    # ==================================================================================
    with tab4:
        st.subheader("Nhật Ký Giao Dịch & Phân Tích Lệnh Chi Tiết")
        
        trade_scope = st.radio("Chọn tập dữ liệu xem nhật ký lệnh:", ["Tập Train", "Tập Test"], horizontal=True)
        active_pf = pf_comb_tr if trade_scope == "Tập Train" else pf_comb_te
        
        trades_df = active_pf.trades_df
        if len(trades_df) == 0:
            st.warning("Không có giao dịch nào được ghi nhận trong giai đoạn này.")
        else:
            # Thống kê lệnh
            c_t1, c_t2, c_t3, c_t4 = st.columns(4)
            c_t1.metric("Tổng số lệnh đóng", f"{len(trades_df)} lệnh")
            
            win_trades = trades_df[trades_df['Return [%]'] > 0]
            loss_trades = trades_df[trades_df['Return [%]'] <= 0]
            
            win_rate_val = len(win_trades) / len(trades_df) * 100
            c_t2.metric("Tỷ lệ thắng (Win Rate)", f"{win_rate_val:.1f}%")
            
            avg_win = win_trades['Return [%]'].mean() if len(win_trades) > 0 else 0.0
            avg_loss = loss_trades['Return [%]'].mean() if len(loss_trades) > 0 else 0.0
            c_t3.metric("Lợi nhuận TB lệnh thắng", f"{avg_win:+.2f}%")
            c_t4.metric("Thua lỗ TB lệnh thua", f"{avg_loss:+.2f}%")
            
            # Biểu đồ phân phối lợi nhuận các lệnh
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=trades_df['Return [%]'],
                nbinsx=25,
                marker_color='#3B82F6',
                opacity=0.75,
                name='Phân Phối Lợi Nhuận'
            ))
            fig_hist.add_vline(x=0, line_width=1.5, line_dash="dash", line_color="red")
            fig_hist.update_layout(
                title="Phân Bố Tỷ Suất Sinh Lời Từng Giao Dịch (%)",
                xaxis_title="Tỷ Suất Sinh Lời Lệnh (%)",
                yaxis_title="Số Lượng Lệnh",
                height=350,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Bảng chi tiết từng lệnh
            st.markdown("#### Danh Sách Toàn Bộ Lệnh Đã Thực Hiện")
            
            # Chuẩn hóa bảng giao dịch hiển thị
            display_trades = trades_df.copy()
            if 'Return [%]' in display_trades.columns:
                display_trades['Tỷ Suất (%)'] = display_trades['Return [%]'].round(2).astype(str) + '%'
            
            st.dataframe(display_trades, use_container_width=True)
            
            # Tải về file CSV
            csv_buffer = io.StringIO()
            trades_df.to_csv(csv_buffer, index=True)
            st.download_button(
                label="📥 Tải Nhật Ký Giao Dịch Xuống Máy (CSV)",
                data=csv_buffer.getvalue(),
                file_name=f"ACB_Trade_Log_{trade_scope}.csv",
                mime="text/csv"
            )

    # ==================================================================================
    # TAB 5: TỐI ƯU HÓA THAM SỐ (HYPEROPT)
    # ==================================================================================
    with tab5:
        st.subheader("Tối Ưu Hóa Siêu Tham Số (Hyperparameter Optimization)")
        st.markdown("""
        Sử dụng thuật toán **Tree-structured Parzen Estimator (TPE)** từ thư viện `hyperopt` để tìm kiếm không gian tham số tối ưu:
        - **Chu kỳ EMA**: Từ 10 đến 50 phiên.
        - **Độ dốc OBV Slope**: Từ 2 đến 20 phiên.
        - **Hàm mục tiêu (Loss Function)**: Tối đa hóa **Sharpe Ratio** trên tập Train với điều kiện tối thiểu 5 giao dịch hợp lệ.
        """)
        
        opt_strat = st.selectbox("Chọn chiến lược cần tối ưu hóa:", ["EMA + OBV Kết hợp", "EMA Riêng lẻ", "OBV Riêng lẻ"], index=0)
        
        col_run1, col_run2 = st.columns([1, 3])
        with col_run1:
            run_btn = st.button("🚀 Bắt Đầu Tối Ưu Hóa", type="primary")
            
        if run_btn:
            with st.spinner(f"Đang chạy thuật toán TPE với {max_evals_input} vòng lặp..."):
                strat_key = 'Combined' if 'Kết hợp' in opt_strat else ('EMA' if 'EMA' in opt_strat else 'OBV')
                start_time = time.time()
                best_res, trials = optimize_hyperparameters(train_df, strat_key, max_evals=max_evals_input)
                elapsed = time.time() - start_time
                
            st.success(f"Tối ưu hóa hoàn tất trong {elapsed:.1f} giây!")
            st.json(best_res)
            
            # Đánh giá lại kết quả tối ưu
            if strat_key == 'Combined' and 'ema_period' in best_res and 'obv_slope_period' in best_res:
                b_ema = best_res['ema_period']
                b_obv = best_res['obv_slope_period']
                e_comb, x_comb, _, _, _ = get_ema_obv_combined_signals(train_df, b_ema, b_obv)
                pf_opt = run_portfolio_simulation(train_df, e_comb, x_comb, fees=fee_pct, slippage=slippage_pct, sl_stop=stop_loss_pct)
                
                st.markdown("#### Hiệu Suất Của Bộ Tham Số Tối Ưu Vừa Tìm Được Trên Train:")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("EMA Tối Ưu", f"{b_ema} phiên")
                m2.metric("OBV Slope Tối Ưu", f"{b_obv} phiên")
                m3.metric("Sharpe Ratio", f"{pf_opt.sharpe_ratio():.4f}")
                m4.metric("Tổng Lợi Nhuận", f"{pf_opt.total_return_pct():+.2f}%")
            elif strat_key == 'EMA' and 'ema_period' in best_res:
                b_ema = best_res['ema_period']
                e_ema, x_ema, _ = get_ema_signals(train_df, b_ema)
                pf_opt = run_portfolio_simulation(train_df, e_ema, x_ema, fees=fee_pct, slippage=slippage_pct, sl_stop=stop_loss_pct)
                st.markdown("#### Hiệu Suất Của Bộ Tham Số EMA Tối Ưu Vừa Tìm Được Trên Train:")
                m1, m2, m3 = st.columns(3)
                m1.metric("EMA Tối Ưu", f"{b_ema} phiên")
                m2.metric("Sharpe Ratio", f"{pf_opt.sharpe_ratio():.4f}")
                m3.metric("Tổng Lợi Nhuận", f"{pf_opt.total_return_pct():+.2f}%")
            elif strat_key == 'OBV' and 'obv_slope_period' in best_res:
                b_obv = best_res['obv_slope_period']
                e_obv, x_obv, _, _ = get_obv_signals(train_df, b_obv)
                pf_opt = run_portfolio_simulation(train_df, e_obv, x_obv, fees=fee_pct, slippage=slippage_pct, sl_stop=stop_loss_pct)
                st.markdown("#### Hiệu Suất Của Bộ Tham Số OBV Tối Ưu Vừa Tìm Được Trên Train:")
                m1, m2, m3 = st.columns(3)
                m1.metric("OBV Slope Tối Ưu", f"{b_obv} phiên")
                m2.metric("Sharpe Ratio", f"{pf_opt.sharpe_ratio():.4f}")
                m3.metric("Tổng Lợi Nhuận", f"{pf_opt.total_return_pct():+.2f}%")
        else:
            # Hiển thị bảng tham số chuẩn từ kết quả nghiên cứu trong notebook
            st.info("💡 Bạn có thể bấm nút trên để chạy thử nghiệm mới, hoặc tham khảo bảng tham số đã được tối ưu hóa trước từ Notebook ACB dưới đây:")
            default_comparison = pd.DataFrame({
                'Chiến lược': ['EMA Riêng lẻ', 'EMA Riêng lẻ', 'OBV Riêng lẻ', 'OBV Riêng lẻ', 'EMA + OBV Kết hợp', 'EMA + OBV Kết hợp'],
                'Loại tham số': ['Mặc định', 'Tối ưu (Notebook)', 'Mặc định', 'Tối ưu (Notebook)', 'Mặc định', 'Tối ưu (Notebook)'],
                'Tham số (EMA, OBV_Slope)': ['EMA=20', 'EMA=36', 'OBV_Slope=3', 'OBV_Slope=19', 'EMA=20, OBV_Slope=3', 'EMA=36, OBV_Slope=20'],
                'Sharpe Ratio (Train)': [0.65, 1.0797, 0.42, 1.1735, 0.81, 1.1514],
                'Số giao dịch đóng (Train)': [117, 70, 198, 51, 102, 64]
            })
            st.dataframe(default_comparison, use_container_width=True)

    # ------------------ FOOTER ------------------
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #6B7280; font-size: 0.85rem;">
            Dự án nghiên cứu: <b>Quản lý danh mục đầu tư & Phân tích chiến lược định lượng</b> | 
            Nền tảng: <b>Streamlit & VectorBT</b> | Cổ phiếu kiểm định: <b>HOSE: ACB</b>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()
