# 📈 Web App Kiểm Định Chiến Lược Giao Dịch: Kết Hợp EMA & Độ Dốc OBV (Cổ Phiếu ACB)

Ứng dụng web tương tác phân tích định lượng (Quantitative Trading) và kiểm định hiệu quả chiến lược kết hợp giữa **Đường Trung Bình Lũy Thừa (EMA)** và **Độ Dốc Khối Lượng Cân Bằng (OBV Slope)** trên dữ liệu lịch sử giá cổ phiếu **ACB (Ngân hàng TMCP Á Châu)** từ năm 2014 đến 2023.

---

## 🌟 Tính Năng Nổi Bật Của Ứng Dụng

- **So sánh 3 Chiến lược Kỹ thuật**:
  1. *Chiến lược EMA Riêng lẻ* (Trend-following)
  2. *Chiến lược Độ dốc OBV Riêng lẻ* (Volume flow)
  3. *Chiến lược Kết hợp EMA + OBV* (Xác nhận đa chỉ báo)
  4. *Chuẩn so sánh (Benchmark)*: Chiến lược Mua & Nắm giữ (Buy & Hold)
- **Kiểm định Mù (Hold-out Validation)**: Tách biệt hoàn toàn tập **Train (2014 - 2020)** để tối ưu tham số và tập **Test (2021 - 2023)** để kiểm định tính ổn định và phát hiện hiện tượng quá khớp (Overfitting).
- **Phòng chống Thiên lệch Nhìn trước (Look-Ahead Bias)**: Tất cả tín hiệu kỹ thuật đều được áp dụng cơ chế dịch chuyển phiên (`shift(1)`) để phản ánh đúng thực tế giao dịch.
- **Mô phỏng Giao dịch Thực tế**: Tích hợp đầy đủ chi phí giao dịch (0.2%), trượt giá (0.1%) và cơ chế cắt lỗ bắt buộc (**Stop-Loss 7%**).
- **Trực quan hóa Tương tác (Plotly)**: Biểu đồ nến kỹ thuật, điểm mua/bán đa khung, đường cong tăng trưởng vốn (Equity Curves) và biểu đồ phân bổ sụt giảm (Drawdown).
- **Tối ưu hóa Siêu tham số (Hyperopt)**: Tích hợp thuật toán Bayes TPE (*Tree-structured Parzen Estimator*) để tìm kiếm bộ tham số tối ưu Sharpe Ratio.
- **Khuyến nghị Trực tiếp**: Thẻ khuyến nghị trạng thái giao dịch phiên hiện tại (MUA, BÁN, NẮM GIỮ hoặc ĐỨNG NGOÀI).

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
├── app.py                                    # Mã nguồn chính của ứng dụng Streamlit
├── requirements.txt                          # Danh sách các thư viện phụ thuộc
├── README.md                                 # Tài liệu hướng dẫn chi tiết
├── ACB.csv                                   # Dữ liệu lịch sử giá cổ phiếu ACB (2014 - 2023)
└── ACB_Strategy_EMA_OBV_Nhom3_Adjusted.ipynb # Jupyter Notebook nghiên cứu gốc
```

---

## 🧠 Cơ Sở Lý Thuyết & Quy Tắc Chiến Lược

### 1. Đường Trung Bình Động Lũy Thừa (EMA)
$$EMA_t = Close_t \times \alpha + EMA_{t-1} \times (1 - \alpha), \quad \text{với } \alpha = \frac{2}{N + 1}$$
- **Quy tắc**: Mua khi giá đóng cửa phiên trước vượt lên trên đường EMA ($Close_{t-1} > EMA_{t-1}$); Bán khi giá cắt xuống dưới đường EMA ($Close_{t-1} < EMA_{t-1}$).

### 2. Chỉ Báo Khối Lượng Cân Bằng (OBV) & Độ Dốc OBV (Slope)
$$OBV_t = \begin{cases} 
OBV_{t-1} + Volume_t & \text{nếu } Close_t > Close_{t-1} \\ 
OBV_{t-1} - Volume_t & \text{nếu } Close_t < Close_{t-1} \\ 
OBV_{t-1} & \text{nếu } Close_t = Close_{t-1} 
\end{cases}$$
$$\text{OBV\_Slope}_t = OBV_t - OBV_{t - K}$$
- **Quy tắc**: Mua khi áp lực dòng tiền tăng trưởng ($\text{OBV\_Slope}_{t-1} > 0$); Bán khi áp lực dòng tiền suy yếu ($\text{OBV\_Slope}_{t-1} < 0$).

### 3. Chiến Lược Kết Hợp EMA & OBV
- **Điểm vào lệnh (Entry)**: Giá trên đường EMA **VÀ** Độ dốc OBV dương $\rightarrow$ Đảm bảo xu hướng giá tăng được xác nhận bởi dòng tiền thực.
- **Điểm thoát lệnh (Exit)**: Giá cắt xuống dưới đường EMA **HOẶC** kích hoạt ngưỡng cắt lỗ **Stop-Loss 7%**.

---

## 📊 Tóm Tắt Kết Quả Nghiên Cứu (ACB)

| Chiến lược | Tập Dữ Liệu | Tham số | Sharpe Ratio | Tổng Lợi Nhuận (%) | Max Drawdown (%) | Số Lệnh Đóng |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EMA Riêng lẻ** | Train (2014-20) | EMA=36 | 1.0797 | +211.97% | -28.69% | 70 |
| **EMA Riêng lẻ** | Test (2021-23) | EMA=36 | -0.8601 | -37.69% | -59.19% | 41 |
| **OBV Riêng lẻ** | Train (2014-20) | OBV=19 | 1.1735 | +277.80% | -30.46% | 51 |
| **OBV Riêng lẻ** | Test (2021-23) | OBV=19 | -0.6135 | -38.73% | -62.50% | 38 |
| **EMA + OBV Kết Hợp** | Train (2014-20) | EMA=36, OBV=20 | **1.1514** | **+236.56%** | **-23.87%** | 64 |
| **EMA + OBV Kết Hợp** | Test (2021-23) | EMA=36, OBV=20 | -0.8298 | -36.47% | -60.14% | 37 |

### Nhận định Học thuật:
1. **Hiện tượng Quá Khớp (Overfitting)**: Bộ tham số tối ưu đạt kết quả vượt trội trong giai đoạn Uptrend (2014 - 2020) nhưng chịu tổn thất lớn khi thị trường chuyển sang trạng thái biến động mạnh và suy thoái trong chu kỳ tăng lãi suất (2021 - 2023).
2. **Vai trò Của OBV**: Sự kết hợp với OBV giúp giảm bớt số lệnh vào sai (nhiễu tín hiệu) và cải thiện mức sụt giảm tối đa (*Max Drawdown giảm từ -28.69% xuống -23.87% trên tập Train*).

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Cục Bộ (Local)

### Bước 1: Clone kho lưu trữ
```bash
git clone <URL_REPO_GITHUB_CUA_BAN>
cd <TEN_THU_MUC_REPO>
```

### Bước 2: Tạo môi trường ảo Python (Khuyến nghị Python 3.10 hoặc 3.11)
```bash
python -m venv venv
# Kích hoạt trên Windows:
.\venv\Scripts\activate
# Kích hoạt trên macOS/Linux:
source venv/bin/activate
```

### Bước 3: Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

### Bước 4: Khởi chạy ứng dụng Streamlit
```bash
streamlit run app.py
```
Trình duyệt web sẽ tự động mở tại địa chỉ: `http://localhost:8501`.

---

## ☁️ Hướng Dẫn Triển Khai Lên Streamlit Cloud (Miễn Phí)

Để đưa ứng dụng lên internet cho giảng viên và hội đồng đánh giá:

1. **Đưa mã nguồn lên GitHub**:
   - Khởi tạo git và push toàn bộ thư mục (gồm `app.py`, `requirements.txt`, `README.md`, `ACB.csv`) lên repository GitHub cá nhân (đặt ở chế độ *Public*).
   ```bash
   git init
   git add app.py requirements.txt README.md ACB.csv
   git commit -m "Khoi tao Web App Kiem dinh Chien luoc EMA + OBV"
   git branch -M main
   git remote add origin https://github.com/<tai-khoan-cua-ban>/<ten-repo>.git
   git push -u origin main
   ```

2. **Kết nối với Streamlit Community Cloud**:
   - Truy cập: [share.streamlit.io](https://share.streamlit.io) và đăng nhập bằng tài khoản **GitHub**.
   - Bấm nút **"New app"** (hoặc "Create app").
   - Điền thông tin:
     - **Repository**: Chọn repo vừa tạo ở bước 1.
     - **Branch**: `main`
     - **Main file path**: `app.py`
   - Bấm **"Deploy!"**.

3. **Hoàn tất**: Sau 1 - 2 phút, Streamlit Cloud sẽ tự động cài đặt các gói trong `requirements.txt` và cung cấp đường dẫn web công khai (ví dụ: `https://your-app-name.streamlit.app`).

---

## 🛠️ Công Nghệ Sử Dụng

- **Ngôn ngữ**: Python 3.10+
- **Giao diện & Ứng dụng Web**: Streamlit
- **Mô phỏng Danh mục Đầu tư**: VectorBT, NumPy, Pandas
- **Chỉ báo Kỹ thuật**: TA (Technical Analysis Library)
- **Tối ưu hóa Siêu tham số**: Hyperopt (TPE Algorithm)
- **Đồ thị Tương tác**: Plotly

---

## 👥 Tác Giả & Học Phần
- **Học phần**: Quản lý danh mục đầu tư (MFB025A)
- **Trường**: Đại học Mở TP.HCM (Open University)
