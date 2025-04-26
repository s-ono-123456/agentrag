import streamlit as st
import pandas as pd
import numpy as np

# ページ設定
st.set_page_config(
    page_title="シンプルなStreamlitアプリ",
    page_icon="📊",
    layout="wide"
)

# タイトルと説明
st.title("シンプルなStreamlitアプリ")
st.subheader("Streamlitで作成した簡単なデータ可視化アプリケーション")

# サイドバー
st.sidebar.header("設定")
sample_size = st.sidebar.slider("サンプルサイズ", 10, 1000, 100)
chart_type = st.sidebar.selectbox("チャートタイプ", ["折れ線グラフ", "棒グラフ", "散布図"])

# データの生成
if st.sidebar.button("データ生成"):
    st.session_state.data = pd.DataFrame({
        'x': np.arange(sample_size),
        'y': np.random.randn(sample_size).cumsum()
    })

# データがない場合は初期データを生成
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame({
        'x': np.arange(100),
        'y': np.random.randn(100).cumsum()
    })

# データ表示
st.subheader("生成されたデータ")
st.dataframe(st.session_state.data)

# チャート表示
st.subheader("データの可視化")
if chart_type == "折れ線グラフ":
    st.line_chart(st.session_state.data.set_index('x'))
elif chart_type == "棒グラフ":
    st.bar_chart(st.session_state.data.set_index('x'))
else:
    st.scatter_chart(st.session_state.data.set_index('x'))

# 統計情報
st.subheader("データの統計情報")
st.write(st.session_state.data.describe())

# フッター
st.markdown("---")
st.caption("© 2025 シンプルなStreamlitアプリ")