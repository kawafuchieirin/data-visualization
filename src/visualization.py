"""
可視化モジュール - グラフとチャート生成
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import streamlit as st

from src.config import Config


class Visualizer:
    """
    データ可視化を担当するクラス
    """
    
    def __init__(self):
        """
        ビジュアライザーを初期化
        """
        self.color_schemes = Config.COLOR_SCHEMES
        self.weather_jp_map = {
            'Clear': '晴れ',
            'Rain': '雨',
            'Clouds': '曇り',
            'Drizzle': '霧雨',
            'Thunderstorm': '雷雨',
            'Snow': '雪',
            'Mist': '霧',
            'Fog': '濃霧'
        }
    
    def create_delay_by_weather_bar(self, data: pd.DataFrame) -> go.Figure:
        """
        天候別平均遅延時間の棒グラフを作成
        
        Args:
            data: 分析済みデータ
            
        Returns:
            go.Figure: Plotlyのフィギュア
        """
        weather_delay = data.groupby('weather_main')['delay_minutes'].mean().sort_values(ascending=False)
        
        # 英語から日本語に変換
        weather_labels = [self.weather_jp_map.get(weather, weather) for weather in weather_delay.index]
        
        fig = px.bar(
            x=weather_labels,
            y=weather_delay.values,
            labels={'x': '天候', 'y': '平均遅延時間（分）'},
            color=weather_delay.values,
            color_continuous_scale=self.color_schemes['delay_bar']
        )
        
        fig.update_layout(
            title="天候別の平均遅延時間",
            showlegend=False
        )
        
        return fig
    
    def create_delay_by_line_bar(self, data: pd.DataFrame) -> go.Figure:
        """
        路線別平均遅延時間の横棒グラフを作成
        
        Args:
            data: 分析済みデータ
            
        Returns:
            go.Figure: Plotlyのフィギュー
        """
        line_delay = data.groupby('line')['delay_minutes'].mean().sort_values(ascending=False)
        
        fig = px.bar(
            x=line_delay.values,
            y=line_delay.index,
            orientation='h',
            labels={'x': '平均遅延時間（分）', 'y': '路線'},
            color=line_delay.values,
            color_continuous_scale=self.color_schemes['line_bar']
        )
        
        fig.update_layout(
            title="路線別の平均遅延時間",
            showlegend=False
        )
        
        return fig
    
    def create_delay_heatmap(self, heatmap_data: pd.DataFrame) -> go.Figure:
        """
        路線×天候の遅延ヒートマップを作成
        
        Args:
            heatmap_data: ヒートマップ用データ
            
        Returns:
            go.Figure: Plotlyのフィギュア
        """
        # 天候ラベルを日本語に変換
        weather_labels_jp = [self.weather_jp_map.get(weather, weather) for weather in heatmap_data.columns]
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=weather_labels_jp,
            y=heatmap_data.index,
            colorscale=self.color_schemes['heatmap'],
            text=heatmap_data.values.round(1),
            texttemplate='%{text}分',
            textfont={"size": 10},
            colorbar=dict(title="平均遅延時間（分）")
        ))
        
        fig.update_layout(
            title="路線別・天候別の平均遅延時間",
            xaxis_title="天候",
            yaxis_title="路線",
            height=600
        )
        
        return fig
    
    def create_daily_trend_line(self, data: pd.DataFrame) -> go.Figure:
        """
        日別遅延推移の折れ線グラフを作成
        
        Args:
            data: 分析済みデータ
            
        Returns:
            go.Figure: Plotlyのフィギュア
        """
        daily_delay = data.groupby(data['timestamp'].dt.date)['delay_minutes'].mean()
        
        fig = px.line(
            x=daily_delay.index,
            y=daily_delay.values,
            labels={'x': '日付', 'y': '平均遅延時間（分）'},
            title="過去30日間の遅延推移"
        )
        
        # 移動平均線を追加
        if len(daily_delay) >= 7:
            rolling_mean = daily_delay.rolling(window=7).mean()
            fig.add_scatter(
                x=rolling_mean.index,
                y=rolling_mean.values,
                mode='lines',
                name='7日移動平均',
                line=dict(dash='dash')
            )
        
        return fig
    
    def create_hourly_pattern_bar(self, data: pd.DataFrame) -> go.Figure:
        """
        時間帯別遅延パターンの棒グラフを作成
        
        Args:
            data: 分析済みデータ
            
        Returns:
            go.Figure: Plotlyのフィギュア
        """
        hourly_delay = data.groupby('hour_of_day')['delay_minutes'].mean()
        
        # ラッシュアワーを色分け
        colors = []
        for hour in hourly_delay.index:
            if any(start <= hour <= end for start, end in Config.RUSH_HOURS):
                colors.append('salmon')
            else:
                colors.append('lightblue')
        
        fig = go.Figure(data=[
            go.Bar(
                x=hourly_delay.index,
                y=hourly_delay.values,
                marker_color=colors
            )
        ])
        
        fig.update_layout(
            title="24時間の遅延パターン（ラッシュアワー：赤）",
            xaxis=dict(
                title="時間",
                tickmode='linear',
                tick0=0,
                dtick=1
            ),
            yaxis_title="平均遅延時間（分）"
        )
        
        return fig
    
    def create_weekday_pattern_bar(self, data: pd.DataFrame) -> go.Figure:
        """
        曜日別遅延パターンの棒グラフを作成
        
        Args:
            data: 分析済みデータ
            
        Returns:
            go.Figure: Plotlyのフィギュア
        """
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        days_japanese = ['月', '火', '水', '木', '金', '土', '日']
        
        weekday_delay = data.groupby('day_of_week')['delay_minutes'].mean().reindex(days_order)
        
        fig = px.bar(
            x=days_japanese,
            y=weekday_delay.values,
            labels={'x': '曜日', 'y': '平均遅延時間（分）'},
            color=weekday_delay.values,
            color_continuous_scale=self.color_schemes['time_series']
        )
        
        fig.update_layout(
            title="曜日別の平均遅延時間",
            showlegend=False
        )
        
        return fig
    
    def create_correlation_scatter(self, data: pd.DataFrame, x_col: str, y_col: str) -> go.Figure:
        """
        相関散布図を作成
        
        Args:
            data: 分析済みデータ
            x_col: X軸のカラム名
            y_col: Y軸のカラム名
            
        Returns:
            go.Figure: Plotlyのフィギュア
        """
        # データの前処理（欠損値や無限値を除去）
        clean_data = data[[x_col, y_col, 'weather_main']].dropna()
        clean_data = clean_data[
            (clean_data[x_col] != float('inf')) & 
            (clean_data[y_col] != float('inf')) &
            (clean_data[x_col] != float('-inf')) & 
            (clean_data[y_col] != float('-inf'))
        ]
        
        # 天候を日本語に変換
        clean_data = clean_data.copy()
        clean_data['weather_jp'] = clean_data['weather_main'].map(self.weather_jp_map).fillna(clean_data['weather_main'])
        
        if clean_data.empty:
            # データが空の場合は空のグラフを返す
            fig = go.Figure()
            fig.update_layout(
                title=f"{self._get_japanese_label(x_col)} vs {self._get_japanese_label(y_col)}",
                xaxis_title=self._get_japanese_label(x_col),
                yaxis_title=self._get_japanese_label(y_col),
                annotations=[{
                    'text': 'データが利用できません',
                    'x': 0.5,
                    'y': 0.5,
                    'xref': 'paper',
                    'yref': 'paper',
                    'showarrow': False,
                    'font': {'size': 16}
                }]
            )
            return fig
        
        fig = px.scatter(
            clean_data,
            x=x_col,
            y=y_col,
            color='weather_jp',
            labels={
                x_col: self._get_japanese_label(x_col),
                y_col: self._get_japanese_label(y_col),
                'weather_jp': '天候'
            },
            title=f"{self._get_japanese_label(x_col)} vs {self._get_japanese_label(y_col)}"
        )
        
        return fig
    
    def create_hourly_weather_heatmap(self, heatmap_data: pd.DataFrame) -> go.Figure:
        """
        時間帯×天候の遅延ヒートマップを作成
        
        Args:
            heatmap_data: 時間帯×天候のデータ
            
        Returns:
            go.Figure: Plotlyのフィギュア
        """
        # 天候ラベルを日本語に変換
        weather_labels_jp = [self.weather_jp_map.get(weather, weather) for weather in heatmap_data.columns]
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=weather_labels_jp,
            y=[f"{hour:02d}時" for hour in heatmap_data.index],
            colorscale=self.color_schemes['heatmap'],
            text=heatmap_data.values.round(1),
            texttemplate='%{text}分',
            textfont={"size": 10},
            colorbar=dict(title="平均遅延時間（分）")
        ))
        
        fig.update_layout(
            title="時間帯別・天候別の平均遅延時間",
            xaxis_title="天候",
            yaxis_title="時間帯",
            height=600
        )
        
        return fig
    
    def create_weather_time_trend(self, data: pd.DataFrame) -> go.Figure:
        """
        天候別の時間推移グラフを作成
        
        Args:
            data: 時間帯と天候別のデータ
            
        Returns:
            go.Figure: Plotlyのフィギュア
        """
        fig = go.Figure()
        
        for weather in data['weather_main'].unique():
            weather_data = data[data['weather_main'] == weather]
            weather_jp = self.weather_jp_map.get(weather, weather)
            
            fig.add_trace(go.Scatter(
                x=weather_data['hour_of_day'],
                y=weather_data['delay_minutes'],
                mode='lines+markers',
                name=weather_jp,
                line=dict(width=3)
            ))
        
        fig.update_layout(
            title="天候別の時間帯推移",
            xaxis_title="時間帯",
            yaxis_title="平均遅延時間（分）",
            xaxis=dict(tickmode='linear', tick0=0, dtick=2),
            height=400
        )
        
        return fig
    
    def _get_japanese_label(self, col_name: str) -> str:
        """
        カラム名を日本語ラベルに変換
        
        Args:
            col_name: カラム名
            
        Returns:
            str: 日本語ラベル
        """
        label_map = {
            'rain_1h': '降雨量 (mm/h)',
            'delay_minutes': '遅延時間（分）',
            'temperature': '気温（℃）',
            'humidity': '湿度（%）',
            'wind_speed': '風速（m/s）'
        }
        return label_map.get(col_name, col_name)


class MetricsDisplay:
    """
    主要指標の表示を担当するクラス
    """
    
    def __init__(self):
        """
        メトリクス表示クラスを初期化
        """
        self.weather_jp_map = {
            'Clear': '晴れ',
            'Rain': '雨',
            'Clouds': '曇り',
            'Drizzle': '霧雨',
            'Thunderstorm': '雷雨',
            'Snow': '雪',
            'Mist': '霧',
            'Fog': '濃霧'
        }
    
    def show_main_metrics(self, data: pd.DataFrame, correlations: Dict[str, float]):
        """
        主要指標を表示
        
        Args:
            data: 分析済みデータ
            correlations: 相関情報
        """
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_delay = data['delay_minutes'].mean()
            st.metric("平均遅延時間", f"{avg_delay:.1f} 分")
        
        with col2:
            delay_rate = (data['delay_minutes'] > 0).mean() * 100
            st.metric("遅延発生率", f"{delay_rate:.1f} %")
        
        with col3:
            ratio = correlations.get('rainy_vs_clear_ratio', 1.0)
            st.metric("降雨時の遅延増加率", f"{ratio:.1f} 倍")
        
        with col4:
            corr = correlations.get('rain_delay_corr', 0.0)
            st.metric("降雨量との相関係数", f"{corr:.3f}")
    
    def show_top_delays(self, heatmap_data: pd.DataFrame, n: int = 5):
        """
        遅延が多い組み合わせを表示
        
        Args:
            heatmap_data: ヒートマップデータ
            n: 表示する件数
        """
        delay_combinations = []
        
        for line in heatmap_data.index:
            for weather in heatmap_data.columns:
                if pd.notna(heatmap_data.loc[line, weather]):
                    weather_jp = self.weather_jp_map.get(weather, weather)
                    delay_combinations.append({
                        '路線': line,
                        '天候': weather_jp,
                        '平均遅延時間': heatmap_data.loc[line, weather]
                    })
        
        top_delays = pd.DataFrame(delay_combinations).nlargest(n, '平均遅延時間')
        st.dataframe(top_delays, use_container_width=True)