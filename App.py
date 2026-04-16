import os
import warnings
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

warnings.filterwarnings("ignore")

# ==============================
# CONFIG
# ==============================
st.set_page_config(
    page_title="Sistem Prediksi Harga Bawang Merah",
    page_icon="📊",
    layout="wide",
)

# ==============================
# CUSTOM CSS
# ==============================
st.markdown(
    """
    <style>
    .main {
        background-color: #f8fafc;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.5rem;
    }
    .hero {
        padding: 24px 28px;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        color: white;
        margin-bottom: 18px;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.18);
    }
    .hero-title {
        font-size: 32px;
        font-weight: 800;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        font-size: 15px;
        color: #dbeafe;
    }
    .soft-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
    }
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
    }
    .metric-label {
        color: #64748b;
        font-size: 13px;
        margin-bottom: 8px;
    }
    .metric-value {
        color: #0f172a;
        font-size: 24px;
        font-weight: 800;
    }
    .section-title {
        font-size: 20px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 8px;
    }
    .mini-note {
        color: #64748b;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================
# HEADER
# ==============================
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">📊 Sistem Prediksi Harga Bawang Merah</div>
        <div class="hero-subtitle">
            Hybrid ARIMA–SVR berbasis Decision Support System untuk estimasi harga komoditas secara interaktif.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================
# HELPERS
# ==============================
def detect_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    date_keywords = ["tahun", "bulan", "tanggal", "date", "periode", "time"]
    price_keywords = ["harga", "price"]

    date_col = next(
        (c for c in df.columns if any(k in c.lower() for k in date_keywords)),
        None,
    )
    price_col = next(
        (c for c in df.columns if any(k in c.lower() for k in price_keywords)),
        None,
    )
    return date_col, price_col


@st.cache_resource(show_spinner=False)
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "hybrid_model_final.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File model tidak ditemukan: {model_path}")

    model = joblib.load(model_path)

    required_keys = {"arima", "svr", "scaler", "residuals", "volatility"}
    if not isinstance(model, dict):
        raise TypeError("Model harus berupa dictionary.")
    if not required_keys.issubset(model.keys()):
        raise KeyError(f"Struktur model tidak lengkap. Keys ditemukan: {list(model.keys())}")

    return model


def load_uploaded_data(uploaded_file) -> pd.DataFrame:
    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Format file tidak didukung. Gunakan CSV atau XLSX.")

    if df.empty:
        raise ValueError("File berhasil dibaca, tetapi isinya kosong.")

    df.columns = df.columns.str.strip().str.lower()
    return df


def run_forecast(
    series: pd.Series,
    arima,
    svr,
    scaler,
    residuals,
    volatility,
    n_future: int,
    seed: int = 42,
) -> pd.Series:
    np.random.seed(seed)

    future_arima = arima.forecast(steps=n_future)

    history_res = list(pd.Series(residuals).dropna().iloc[-12:])
    if len(history_res) < 12:
        raise ValueError("Residual kurang dari 12 observasi. Forecast tidak dapat dijalankan.")

    future_res = []

    for i in range(n_future):
        month = (series.index[-1].month + i) % 12 + 1

        features = history_res[-12:] + [
            month,
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12),
            len(series) + i,
        ]

        features = np.array(features, dtype=float).reshape(1, -1)
        features_scaled = scaler.transform(features)
        svr_pred = float(svr.predict(features_scaled)[0])

        vol = float(volatility[i]) if i < len(volatility) else float(np.mean(volatility))

        dynamic = 0.12 * np.sin(i / 2)
        trend = 0.08 * (i / n_future)
        seasonal = 0.08 * np.sin(2 * np.pi * month / 12)
        noise = np.random.normal(0, max(vol, 0.03))

        final_res = svr_pred + noise + dynamic + trend + seasonal
        final_res = float(np.clip(final_res, -0.4, 0.4))

        future_res.append(final_res)
        history_res.append(final_res)

    hybrid_log = np.asarray(future_arima) + np.array(future_res)
    hybrid = np.exp(hybrid_log)

    future_index = pd.date_range(
        start=series.index[-1],
        periods=n_future + 1,
        freq="ME",
    )[1:]

    return pd.Series(hybrid, index=future_index, name="Prediksi Harga")


def format_rupiah(x: float) -> str:
    return f"Rp {int(x):,}"


def create_download_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def build_chart(series: pd.Series, forecast_series: pd.Series):
    hist_df = series.reset_index()
    hist_df.columns = ["Tanggal", "Harga"]
    hist_df["Tipe"] = "Historis"

    pred_df = forecast_series.reset_index()
    pred_df.columns = ["Tanggal", "Harga"]
    pred_df["Tipe"] = "Prediksi"

    chart_df = pd.concat([hist_df, pred_df], ignore_index=True)

    color_scale = alt.Scale(
        domain=["Historis", "Prediksi"],
        range=["#1d4ed8", "#dc2626"],
    )

    line = (
        alt.Chart(chart_df)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("Tanggal:T", title="Periode"),
            y=alt.Y("Harga:Q", title="Harga"),
            color=alt.Color("Tipe:N", scale=color_scale),
            tooltip=[
                alt.Tooltip("Tanggal:T", title="Tanggal"),
                alt.Tooltip("Harga:Q", title="Harga", format=",.0f"),
                alt.Tooltip("Tipe:N", title="Tipe"),
            ],
        )
        .properties(height=420)
        .interactive()
    )

    return line


# ==============================
# LOAD MODEL
# ==============================
try:
    model = load_model()
    arima = model["arima"]
    svr = model["svr"]
    scaler = model["scaler"]
    residuals = model["residuals"]
    volatility = model["volatility"]
except Exception as e:
    st.error("❌ Gagal load model.")
    st.exception(e)
    st.stop()

# ==============================
# SIDEBAR
# ==============================
st.sidebar.markdown("## ⚙️ Pengaturan Sistem")
st.sidebar.markdown("Gunakan panel ini untuk mengunggah data dan menjalankan prediksi.")

uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["xlsx", "xls", "csv"])
n_future = st.sidebar.slider("Jumlah Bulan Prediksi", 1, 24, 12)
run_button = st.sidebar.button("🚀 Jalankan Prediksi", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info(
    "Model yang digunakan adalah Hybrid ARIMA–SVR dengan koreksi residual, dinamika musiman, dan penyesuaian volatilitas."
)

# ==============================
# MAIN
# ==============================
if uploaded_file is not None:
    try:
        df = load_uploaded_data(uploaded_file)
        date_col, price_col = detect_columns(df)

        if not date_col or not price_col:
            st.error(
                "❌ Kolom tanggal/periode atau harga tidak ditemukan. Pastikan dataset memiliki kolom seperti 'tahun_bulan' dan 'harga'."
            )
            st.write("Kolom yang terbaca:", list(df.columns))
            st.stop()

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col, price_col]).copy()
        df = df.sort_values(date_col)
        df.set_index(date_col, inplace=True)

        series = df[price_col].astype(float)

        if series.empty or len(series) < 24:
            st.error("❌ Data terlalu sedikit. Minimal disarankan 24 observasi.")
            st.stop()

        st.success("✅ Dataset berhasil dimuat")

        # KPI DATASET
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Jumlah Observasi</div>
                    <div class="metric-value">{len(series)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Harga Minimum</div>
                    <div class="metric-value">{format_rupiah(series.min())}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Harga Maksimum</div>
                    <div class="metric-value">{format_rupiah(series.max())}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Harga Terakhir</div>
                    <div class="metric-value">{format_rupiah(series.iloc[-1])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("")

        tab1, tab2, tab3 = st.tabs(["📈 Dashboard Utama", "📋 Data & Prediksi", "ℹ️ Informasi Model"])

        with tab1:
            left, right = st.columns([2.2, 1])

            with left:
                st.markdown('<div class="section-title">Visualisasi Historis Data</div>', unsafe_allow_html=True)
                st.line_chart(series)

            with right:
                st.markdown('<div class="section-title">Ringkasan Dataset</div>', unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div class="soft-card">
                        <p><b>Periode Awal:</b><br>{series.index.min().strftime("%Y-%m-%d")}</p>
                        <p><b>Periode Akhir:</b><br>{series.index.max().strftime("%Y-%m-%d")}</p>
                        <p><b>Rata-rata Historis:</b><br>{format_rupiah(series.mean())}</p>
                        <p><b>Standar Deviasi:</b><br>{format_rupiah(series.std())}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with tab2:
            st.markdown('<div class="section-title">Data Akhir yang Digunakan</div>', unsafe_allow_html=True)
            st.dataframe(
                df[[price_col]].tail(12).rename(columns={price_col: "Harga"}),
                use_container_width=True,
            )

        with tab3:
            st.markdown('<div class="section-title">Deskripsi Sistem</div>', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="soft-card">
                Sistem ini menggunakan pendekatan <b>Hybrid ARIMA–SVR</b> untuk prediksi harga bawang merah.
                Komponen ARIMA digunakan untuk menangkap pola linear utama, sedangkan SVR digunakan
                untuk memodelkan residual agar pola non-linear dapat dikoreksi.  
                Selain itu, sistem menambahkan dinamika musiman, tren, dan komponen volatilitas agar
                hasil prediksi tidak terlalu datar dan lebih realistis secara perilaku pasar.
                </div>
                """,
                unsafe_allow_html=True,
            )

    except Exception as e:
        st.error("❌ Gagal membaca dataset.")
        st.exception(e)
        st.stop()

    if run_button:
        try:
            forecast_series = run_forecast(
                series=series,
                arima=arima,
                svr=svr,
                scaler=scaler,
                residuals=residuals,
                volatility=volatility,
                n_future=n_future,
            )

            st.markdown("---")
            st.markdown('<div class="section-title">Hasil Prediksi</div>', unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Harga Terakhir</div>
                        <div class="metric-value">{format_rupiah(series.iloc[-1])}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Rata-rata Prediksi</div>
                        <div class="metric-value">{format_rupiah(forecast_series.mean())}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Harga Minimum Prediksi</div>
                        <div class="metric-value">{format_rupiah(forecast_series.min())}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m4:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Harga Maksimum Prediksi</div>
                        <div class="metric-value">{format_rupiah(forecast_series.max())}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("")

            chart = build_chart(series, forecast_series)
            st.altair_chart(chart, use_container_width=True)

            # Insight otomatis
            delta_mean = forecast_series.mean() - series.iloc[-1]
            direction = "naik" if delta_mean > 0 else "turun"

            c_left, c_right = st.columns([1.4, 1])

            with c_left:
                st.markdown('<div class="section-title">Tabel Hasil Prediksi</div>', unsafe_allow_html=True)
                result = forecast_series.reset_index()
                result.columns = ["Tanggal", "Prediksi Harga"]
                st.dataframe(result, use_container_width=True)

                csv_data = create_download_csv(result)
                st.download_button(
                    label="⬇️ Download Hasil Prediksi (CSV)",
                    data=csv_data,
                    file_name="hasil_prediksi_bawang_merah.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with c_right:
                st.markdown('<div class="section-title">Insight Otomatis</div>', unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div class="soft-card">
                        <p><b>Arah Prediksi Rata-rata:</b> {direction}</p>
                        <p><b>Selisih terhadap harga terakhir:</b> {format_rupiah(abs(delta_mean))}</p>
                        <p><b>Horizon Prediksi:</b> {n_future} bulan</p>
                        <p><b>Indikasi Volatilitas:</b> {"Tinggi" if forecast_series.std() > series.std()*0.5 else "Moderat"}</p>
                        <p class="mini-note">
                            Insight ini bersifat deskriptif untuk membantu interpretasi hasil prediksi.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        except Exception as e:
            st.error("❌ Forecast gagal dijalankan.")
            st.exception(e)

else:
    st.info("📂 Silakan upload dataset melalui sidebar untuk memulai proses prediksi.")

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.caption("© 2026 | Sistem Informasi Prediktif - Tesis Magister Sistem Informasi UNDIP")