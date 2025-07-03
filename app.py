"""
電車遅延×天気データ相関分析 Streamlitアプリケーション
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import sys

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_analyzer import TrainDelayAnalyzer
from src.train_delay_collector import TrainDelayCollector
from src.weather_collector import WeatherCollector
from src.visualization import Visualizer, MetricsDisplay
from src.config import Config

# ページ設定
st.set_page_config(
    page_title=Config.PAGE_TITLE,
    page_icon=Config.PAGE_ICON,
    layout=Config.LAYOUT
)


class TrainDelayDashboard:
    """
    電車遅延ダッシュボードのメインクラス
    """
    
    def __init__(self):
        """
        ダッシュボードを初期化
        """
        self.analyzer = TrainDelayAnalyzer()
        self.visualizer = Visualizer()
        self.metrics_display = MetricsDisplay()
        
    @st.cache_data
    def load_and_analyze_data(_self):
        """
        データの読み込みと分析を実行
        
        Returns:
            tuple: アナライザー、統計、相関、ヒートマップデータ
        """
        analyzer = TrainDelayAnalyzer()
        
        # サンプルデータがない場合は生成
        delay_file = os.path.join(Config.RAW_DATA_DIR, 'sample_train_delays.csv')
        weather_file = os.path.join(Config.RAW_DATA_DIR, 'sample_weather_data.csv')
        
        if not os.path.exists(delay_file) or not os.path.exists(weather_file):
            analyzer.generate_sample_data()
        
        # データ読み込みと分析
        analyzer.load_data(delay_file, weather_file)
        analyzer.merge_data()
        
        # 統計計算
        stats = analyzer.calculate_delay_statistics()
        correlations = analyzer.calculate_correlation()
        heatmap_data = analyzer.create_heatmap_data()
        report = analyzer.generate_summary_report()
        
        return analyzer, stats, correlations, heatmap_data, report
    
    def setup_sidebar(self, merged_df: pd.DataFrame) -> tuple:
        """
        サイドバーのフィルター設定
        
        Args:
            merged_df: マージ済みデータ
            
        Returns:
            tuple: 選択された路線と天候
        """
        st.sidebar.header("フィルター設定")
        
        # 山手線のみなので路線選択は固定
        all_lines = sorted(merged_df['line'].unique())
        selected_lines = all_lines  # 全て選択（山手線のみ）
        
        # 山手線専用の情報を表示
        st.sidebar.info("🚃 **分析対象**: 山手線専用")
        
        # 山手線の基本情報
        st.sidebar.markdown("""
        ### 📊 山手線について
        - **路線長**: 34.5km
        - **駅数**: 30駅
        - **運行間隔**: 3-4分間隔
        - **1日平均利用者数**: 約390万人
        """)
        
        st.sidebar.markdown("---")
        
        # 天候選択
        weather_jp_map = {
            'Clear': '晴れ',
            'Rain': '雨',
            'Clouds': '曇り',
            'Drizzle': '霧雨',
            'Thunderstorm': '雷雨',
            'Snow': '雪',
            'Mist': '霧',
            'Fog': '濃霧'
        }
        
        all_weather = sorted(merged_df['weather_main'].unique())
        weather_options_jp = [weather_jp_map.get(w, w) for w in all_weather]
        
        selected_weather_jp = st.sidebar.multiselect(
            "天候を選択",
            options=weather_options_jp,
            default=weather_options_jp,
            help="分析対象の天候を選択してください"
        )
        
        # 日本語から英語に逆変換
        weather_jp_to_en = {v: k for k, v in weather_jp_map.items()}
        selected_weather = [weather_jp_to_en.get(w, w) for w in selected_weather_jp]
        
        # 期間選択
        st.sidebar.subheader("分析期間")
        date_range = st.sidebar.date_input(
            "期間を選択",
            value=(merged_df['timestamp'].min(), merged_df['timestamp'].max()),
            min_value=merged_df['timestamp'].min(),
            max_value=merged_df['timestamp'].max()
        )
        
        return selected_lines, selected_weather, date_range
    
    def filter_data(self, 
                   merged_df: pd.DataFrame, 
                   selected_lines: list, 
                   selected_weather: list,
                   date_range: tuple) -> pd.DataFrame:
        """
        データをフィルタリング
        
        Args:
            merged_df: マージ済みデータ
            selected_lines: 選択された路線
            selected_weather: 選択された天候
            date_range: 選択された期間
            
        Returns:
            pd.DataFrame: フィルタリング済みデータ
        """
        filtered_df = merged_df[
            (merged_df['line'].isin(selected_lines)) &
            (merged_df['weather_main'].isin(selected_weather))
        ]
        
        # 期間でフィルタリング
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = filtered_df[
                (filtered_df['timestamp'].dt.date >= start_date) &
                (filtered_df['timestamp'].dt.date <= end_date)
            ]
        
        return filtered_df
    
    def show_overview_tab(self, filtered_df: pd.DataFrame, correlations: dict):
        """
        概要タブの表示
        
        Args:
            filtered_df: フィルタリング済みデータ
            correlations: 相関情報
        """
        st.header("📊 遅延統計の概要")
        
        # 主要指標
        self.metrics_display.show_main_metrics(filtered_df, correlations)
        
        # 天候別遅延グラフ（山手線専用）
        st.subheader("山手線の天候別平均遅延時間")
        fig = self.visualizer.create_delay_by_weather_bar(filtered_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # 山手線専用の追加分析
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("ラッシュアワー vs オフピーク")
            rush_hours = [7, 8, 9, 17, 18, 19, 20]
            rush_data = filtered_df[filtered_df['hour_of_day'].isin(rush_hours)]
            off_peak_data = filtered_df[~filtered_df['hour_of_day'].isin(rush_hours)]
            
            comparison_data = pd.DataFrame({
                '時間帯': ['ラッシュアワー', 'オフピーク'],
                '平均遅延時間': [rush_data['delay_minutes'].mean(), off_peak_data['delay_minutes'].mean()]
            })
            
            fig_rush = px.bar(
                comparison_data,
                x='時間帯',
                y='平均遅延時間',
                color='平均遅延時間',
                color_continuous_scale='Reds'
            )
            fig_rush.update_layout(showlegend=False)
            st.plotly_chart(fig_rush, use_container_width=True)
        
        with col2:
            st.subheader("雨天時の時間帯別影響")
            rainy_data = filtered_df[filtered_df['is_rainy'] == True]
            if not rainy_data.empty:
                hourly_rain_impact = rainy_data.groupby('hour_of_day')['delay_minutes'].mean()
                
                fig_rain_hour = px.line(
                    x=hourly_rain_impact.index,
                    y=hourly_rain_impact.values,
                    labels={'x': '時間帯', 'y': '平均遅延時間（分）'},
                    markers=True
                )
                fig_rain_hour.update_layout(showlegend=False)
                st.plotly_chart(fig_rain_hour, use_container_width=True)
            else:
                st.info("雨天データがありません")
    
    def show_heatmap_tab(self, filtered_df: pd.DataFrame):
        """
        時間帯×天候分析タブの表示（山手線専用）
        
        Args:
            filtered_df: フィルタリング済みデータ
        """
        st.header("🕐 時間帯×天候 詳細分析（山手線専用）")
        
        # 時間帯×天候のヒートマップデータ準備
        heatmap_data = filtered_df.pivot_table(
            values='delay_minutes',
            index='hour_of_day',
            columns='weather_main',
            aggfunc='mean',
            fill_value=0
        ).round(1)
        
        # 時間帯×天候のヒートマップ表示
        fig = self.visualizer.create_hourly_weather_heatmap(heatmap_data)
        st.plotly_chart(fig, use_container_width=True)
        
        # 山手線の特徴的な時間帯分析
        st.subheader("📊 山手線の時間帯別特徴")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # ラッシュアワーでの天候影響
            rush_hours = [7, 8, 9, 17, 18, 19, 20]
            rush_data = filtered_df[filtered_df['hour_of_day'].isin(rush_hours)]
            rush_weather_impact = rush_data.groupby('weather_main')['delay_minutes'].mean().sort_values(ascending=False)
            
            st.markdown("**ラッシュアワーでの天候影響**")
            weather_jp_map = {
                'Clear': '晴れ', 'Rain': '雨', 'Clouds': '曇り', 
                'Drizzle': '霧雨', 'Thunderstorm': '雷雨', 'Snow': '雪'
            }
            
            for weather, delay in rush_weather_impact.items():
                weather_jp = weather_jp_map.get(weather, weather)
                st.write(f"- {weather_jp}: {delay:.1f}分")
        
        with col2:
            # 最も遅延が多い時間帯TOP5
            hourly_delays = filtered_df.groupby('hour_of_day')['delay_minutes'].mean().sort_values(ascending=False).head(5)
            
            st.markdown("**最も遅延が多い時間帯 TOP5**")
            for hour, delay in hourly_delays.items():
                period = "朝ラッシュ" if 7 <= hour <= 9 else "夕ラッシュ" if 17 <= hour <= 20 else "オフピーク"
                st.write(f"- {hour:02d}時台: {delay:.1f}分 ({period})")
        
        # 天候別の時間推移
        st.subheader("📈 天候別の時間推移")
        weather_time_data = filtered_df.groupby(['hour_of_day', 'weather_main'])['delay_minutes'].mean().reset_index()
        
        fig_line = self.visualizer.create_weather_time_trend(weather_time_data)
        st.plotly_chart(fig_line, use_container_width=True)
    
    def show_timeseries_tab(self, filtered_df: pd.DataFrame):
        """
        時系列分析タブの表示
        
        Args:
            filtered_df: フィルタリング済みデータ
        """
        st.header("📈 時系列分析")
        
        # 日別遅延推移
        st.subheader("日別の平均遅延時間推移")
        fig = self.visualizer.create_daily_trend_line(filtered_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # 時間帯別と曜日別を横に配置
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("時間帯別の平均遅延時間")
            fig = self.visualizer.create_hourly_pattern_bar(filtered_df)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("曜日別の平均遅延時間")
            fig = self.visualizer.create_weekday_pattern_bar(filtered_df)
            st.plotly_chart(fig, use_container_width=True)
    
    def show_detail_tab(self, filtered_df: pd.DataFrame):
        """
        詳細データタブの表示
        
        Args:
            filtered_df: フィルタリング済みデータ
        """
        st.header("🔍 詳細データ")
        
        # 相関分析
        st.subheader("天候要因との相関分析")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 降雨量と遅延の散布図
            if 'rain_1h' in filtered_df.columns and not filtered_df['rain_1h'].isna().all():
                fig = self.visualizer.create_correlation_scatter(
                    filtered_df, 'rain_1h', 'delay_minutes'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("降雨量データが利用できません")
        
        with col2:
            # 湿度と遅延の散布図
            if 'humidity' in filtered_df.columns and not filtered_df['humidity'].isna().all():
                fig = self.visualizer.create_correlation_scatter(
                    filtered_df, 'humidity', 'delay_minutes'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("湿度データが利用できません")
        
        # データテーブル
        st.subheader("生データ（フィルター適用済み）")
        
        # 表示する列を選択
        display_columns = st.multiselect(
            "表示する列を選択",
            options=filtered_df.columns.tolist(),
            default=['timestamp', 'line', 'delay_minutes', 'weather_main', 
                    'temperature', 'humidity', 'rain_1h', 'reason']
        )
        
        # データ表示
        if display_columns:
            st.dataframe(
                filtered_df[display_columns].sort_values('timestamp', ascending=False).head(100),
                use_container_width=True
            )
        
        # データダウンロード
        csv = filtered_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 CSVダウンロード",
            data=csv,
            file_name=f"train_delay_weather_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # 統計サマリー
        st.subheader("統計サマリー")
        numeric_columns = ['delay_minutes', 'temperature', 'humidity', 'rain_1h', 'wind_speed']
        available_columns = [col for col in numeric_columns if col in filtered_df.columns]
        if available_columns:
            st.write(filtered_df[available_columns].describe().round(2))
        else:
            st.warning("数値データが見つかりません。")
    
    def show_insights_tab(self, report: dict):
        """
        インサイトタブの表示
        
        Args:
            report: 分析レポート
        """
        st.header("💡 分析インサイト")
        
        # データ概要
        st.subheader("📊 データ概要")
        data_overview = report.get('データ概要', {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("総レコード数", f"{data_overview.get('総レコード数', 0):,}")
        with col2:
            st.metric("路線数", data_overview.get('路線数', 0))
        with col3:
            st.metric("天候パターン数", data_overview.get('天候パターン数', 0))
        
        # 分析期間
        if '分析期間' in data_overview:
            period = data_overview['分析期間']
            st.info(f"分析期間: {period.get('開始', 'N/A')} 〜 {period.get('終了', 'N/A')}")
        
        # 主要な発見
        st.subheader("🔍 主要な発見")
        findings = report.get('主要な発見', {})
        
        for key, value in findings.items():
            st.write(f"- **{key}**: {value}")
        
        # 推奨事項
        st.subheader("💡 推奨事項")
        recommendations = report.get('推奨事項', [])
        
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
        
        # 山手線専用の分析提案
        st.subheader("🚀 山手線専用の分析提案")
        st.markdown("""
        - **駅別遅延分析**: 新宿、渋谷、池袋など主要駅での遅延パターン
        - **内回り・外回り別分析**: 運行方向による遅延差の分析
        - **時間帯別詳細分析**: ラッシュ時の天候影響をより詳細に
        - **他路線への波及効果**: 山手線遅延が接続路線に与える影響
        - **特殊イベント影響**: 花火大会、コンサートなど大型イベント時の分析
        """)
        
        # 山手線の特徴的な情報
        st.subheader("🔍 山手線の特徴")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **運行の特徴**
            - 環状運行で始発・終電なし
            - 高頻度運行（3-4分間隔）
            - 都心部の主要駅を結ぶ
            """)
        
        with col2:
            st.markdown("""
            **天候による影響**
            - 高架区間での風の影響
            - 都市部特有の集中豪雨
            - 乗客集中による遅延拡大
            """)
    
    def run(self):
        """
        ダッシュボードを実行
        """
        st.title("🚃 山手線遅延 × 天気データ 相関分析")
        st.markdown("雨の日に山手線がどれだけ遅れるのか？天候と山手線の遅延関係を詳細に分析します。")
        
        # データ読み込み
        with st.spinner("データを読み込んでいます..."):
            analyzer, stats, correlations, heatmap_data, report = self.load_and_analyze_data()
        
        # サイドバー設定
        selected_lines, selected_weather, date_range = self.setup_sidebar(analyzer.merged_df)
        
        # データフィルタリング
        filtered_df = self.filter_data(
            analyzer.merged_df, 
            selected_lines, 
            selected_weather,
            date_range
        )
        
        # データが空の場合の警告
        if filtered_df.empty:
            st.warning("選択された条件に該当するデータがありません。フィルター条件を調整してください。")
            return
        
        # タブ作成（山手線専用）
        tabs = st.tabs([
            "📊 概要", 
            "🕐 時間帯×天候分析", 
            "📈 時系列分析", 
            "🔍 詳細データ",
            "💡 山手線インサイト"
        ])
        
        with tabs[0]:
            self.show_overview_tab(filtered_df, correlations)
        
        with tabs[1]:
            self.show_heatmap_tab(filtered_df)
        
        with tabs[2]:
            self.show_timeseries_tab(filtered_df)
        
        with tabs[3]:
            self.show_detail_tab(filtered_df)
        
        with tabs[4]:
            self.show_insights_tab(report)


def main():
    """
    メイン関数
    """
    # 設定検証
    Config.validate()
    
    # ダッシュボード実行
    dashboard = TrainDelayDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()