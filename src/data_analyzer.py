"""
データ分析モジュール - 遅延と天気の相関分析
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import os
import sys

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config

logging.basicConfig(level=logging.INFO, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class TrainDelayAnalyzer:
    """
    電車遅延と天気データの相関を分析するクラス
    
    Attributes:
        delay_df: 遅延データのDataFrame
        weather_df: 天気データのDataFrame
        merged_df: 結合されたデータのDataFrame
    """
    
    def __init__(self):
        """
        アナライザーを初期化
        """
        self.delay_df: Optional[pd.DataFrame] = None
        self.weather_df: Optional[pd.DataFrame] = None
        self.merged_df: Optional[pd.DataFrame] = None
        self.logger = logging.getLogger(__name__)
        
    def load_data(self, delay_file: str, weather_file: str) -> None:
        """
        遅延データと天気データを読み込み
        
        Args:
            delay_file: 遅延データファイルのパス
            weather_file: 天気データファイルのパス
            
        Raises:
            FileNotFoundError: ファイルが見つからない場合
            ValueError: データ形式が不正な場合
        """
        try:
            self.delay_df = pd.read_csv(delay_file)
            self.delay_df['timestamp'] = pd.to_datetime(self.delay_df['timestamp'])
            
            self.weather_df = pd.read_csv(weather_file)
            self.weather_df['timestamp'] = pd.to_datetime(self.weather_df['timestamp'])
            
            self.logger.info(
                f"Loaded {len(self.delay_df)} delay records and "
                f"{len(self.weather_df)} weather records"
            )
            
            # データ検証
            self._validate_loaded_data()
            
        except FileNotFoundError as e:
            self.logger.error(f"File not found: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def _validate_loaded_data(self) -> None:
        """
        読み込んだデータの妥当性を検証
        
        Raises:
            ValueError: データが不正な場合
        """
        # 遅延データの検証
        delay_required_cols = ['timestamp', 'line', 'delay_minutes', 'status']
        missing_delay_cols = set(delay_required_cols) - set(self.delay_df.columns)
        if missing_delay_cols:
            raise ValueError(f"Missing delay columns: {missing_delay_cols}")
        
        # 天気データの検証
        weather_required_cols = ['timestamp', 'weather_main', 'temperature', 'humidity']
        missing_weather_cols = set(weather_required_cols) - set(self.weather_df.columns)
        if missing_weather_cols:
            raise ValueError(f"Missing weather columns: {missing_weather_cols}")
    
    def merge_data(self) -> pd.DataFrame:
        """
        遅延データと天気データを時間でマージ
        
        Returns:
            pd.DataFrame: マージされたデータ
            
        Raises:
            ValueError: データが読み込まれていない場合
        """
        if self.delay_df is None or self.weather_df is None:
            raise ValueError("Data must be loaded before merging")
        
        # 時間を丸めて結合（1時間単位）
        self.delay_df['hour'] = self.delay_df['timestamp'].dt.floor('h')
        self.weather_df['hour'] = self.weather_df['timestamp'].dt.floor('h')
        
        # 天気データと遅延データを結合
        # 利用可能な天気データカラムを動的に選択
        weather_cols = ['hour', 'weather_main', 'temperature', 'humidity', 'rain_1h', 'is_rainy']
        
        # wind_speedが存在する場合は追加
        if 'wind_speed' in self.weather_df.columns:
            weather_cols.append('wind_speed')
        
        # 実際に存在するカラムのみを使用
        available_weather_cols = [col for col in weather_cols if col in self.weather_df.columns]
        
        self.merged_df = pd.merge(
            self.delay_df,
            self.weather_df[available_weather_cols],
            on='hour',
            how='left'
        )
        
        # 追加の特徴量を生成
        self._add_derived_features()
        
        self.logger.info(f"Merged data contains {len(self.merged_df)} records")
        return self.merged_df
    
    def _add_derived_features(self) -> None:
        """
        派生特徴量を追加
        """
        if self.merged_df is None:
            return
        
        # 時間帯別特徴量
        self.merged_df['hour_of_day'] = self.merged_df['timestamp'].dt.hour
        self.merged_df['day_of_week'] = self.merged_df['timestamp'].dt.day_name()
        self.merged_df['is_weekend'] = self.merged_df['timestamp'].dt.dayofweek.isin([5, 6])
        
        # ラッシュアワーフラグ
        self.merged_df['is_rush_hour'] = self.merged_df['hour_of_day'].apply(
            lambda h: any(start <= h <= end for start, end in Config.RUSH_HOURS)
        )
        
        # 遅延カテゴリ
        self.merged_df['delay_category'] = pd.cut(
            self.merged_df['delay_minutes'],
            bins=[0, 5, 15, 30, float('inf')],
            labels=['正常', '軽微', '中程度', '重度'],
            include_lowest=True
        )
    
    def calculate_delay_statistics(self) -> Dict[str, Any]:
        """
        遅延統計を計算
        
        Returns:
            Dict[str, Any]: 各種統計情報
            
        Raises:
            ValueError: データがマージされていない場合
        """
        if self.merged_df is None:
            raise ValueError("Data must be merged before calculating statistics")
        
        stats = {}
        
        # 路線別の平均遅延時間
        stats['avg_delay_by_line'] = (
            self.merged_df.groupby('line')['delay_minutes']
            .mean()
            .round(2)
            .to_dict()
        )
        
        # 天候別の平均遅延時間
        stats['avg_delay_by_weather'] = (
            self.merged_df.groupby('weather_main')['delay_minutes']
            .mean()
            .round(2)
            .to_dict()
        )
        
        # 路線×天候の遅延統計
        delay_rate = self.merged_df.groupby(['line', 'weather_main']).agg({
            'delay_minutes': ['mean', 'count', lambda x: (x > 0).mean() * 100]
        }).round(2)
        delay_rate.columns = ['平均遅延時間', 'サンプル数', '遅延発生率(%)']
        stats['delay_rate_by_line_weather'] = delay_rate
        
        # 時間帯別遅延傾向
        stats['avg_delay_by_hour'] = (
            self.merged_df.groupby('hour_of_day')['delay_minutes']
            .mean()
            .round(2)
            .to_dict()
        )
        
        # 曜日別遅延傾向
        stats['avg_delay_by_day'] = (
            self.merged_df.groupby('day_of_week')['delay_minutes']
            .mean()
            .round(2)
            .to_dict()
        )
        
        # ラッシュアワー分析
        rush_hour_stats = self.merged_df.groupby('is_rush_hour')['delay_minutes'].agg(['mean', 'std', 'count'])
        stats['rush_hour_analysis'] = rush_hour_stats.to_dict()
        
        return stats
    
    def calculate_correlation(self) -> Dict[str, float]:
        """
        天候要因と遅延の相関を計算
        
        Returns:
            Dict[str, float]: 各種相関係数
            
        Raises:
            ValueError: データがマージされていない場合
        """
        if self.merged_df is None:
            raise ValueError("Data must be merged before calculating correlations")
        
        correlations = {}
        
        # 数値データのみ抽出（存在するカラムのみ）
        numeric_cols = ['delay_minutes', 'rain_1h', 'humidity', 'temperature', 'wind_speed']
        available_cols = [col for col in numeric_cols if col in self.merged_df.columns]
        
        if not available_cols:
            return correlations
            
        numeric_df = self.merged_df[available_cols].dropna()
        
        if not numeric_df.empty:
            # 降雨量と遅延時間の相関
            if 'rain_1h' in numeric_df.columns:
                correlations['rain_delay_corr'] = numeric_df['rain_1h'].corr(numeric_df['delay_minutes'])
            
            # 湿度と遅延時間の相関
            if 'humidity' in numeric_df.columns:
                correlations['humidity_delay_corr'] = numeric_df['humidity'].corr(numeric_df['delay_minutes'])
            
            # 気温と遅延時間の相関
            if 'temperature' in numeric_df.columns:
                correlations['temp_delay_corr'] = numeric_df['temperature'].corr(numeric_df['delay_minutes'])
            
            # 風速と遅延時間の相関
            if 'wind_speed' in numeric_df.columns:
                correlations['wind_delay_corr'] = numeric_df['wind_speed'].corr(numeric_df['delay_minutes'])
        
        # 雨天時vs晴天時の遅延比較
        rainy_delays = self.merged_df[self.merged_df['is_rainy'] == True]['delay_minutes'].mean()
        clear_delays = self.merged_df[self.merged_df['weather_main'] == 'Clear']['delay_minutes'].mean()
        
        if clear_delays > 0:
            correlations['rainy_vs_clear_ratio'] = rainy_delays / clear_delays
        else:
            correlations['rainy_vs_clear_ratio'] = np.inf
        
        # 統計的有意性の検定（簡易版）
        correlations['significant_factors'] = self._test_significance()
        
        return correlations
    
    def _test_significance(self) -> List[str]:
        """
        統計的に有意な要因を特定（簡易版）
        
        Returns:
            List[str]: 有意な要因のリスト
        """
        significant_factors = []
        
        # 天候別の遅延時間の分散分析（簡易版）
        weather_groups = self.merged_df.groupby('weather_main')['delay_minutes'].apply(list)
        
        # グループ間の差が大きいかチェック
        means = {weather: np.mean(delays) for weather, delays in weather_groups.items()}
        overall_mean = self.merged_df['delay_minutes'].mean()
        
        for weather, mean in means.items():
            if abs(mean - overall_mean) > overall_mean * 0.3:  # 30%以上の差
                significant_factors.append(f"{weather}天候")
        
        return significant_factors
    
    def create_heatmap_data(self) -> pd.DataFrame:
        """
        ヒートマップ用のデータを作成
        
        Returns:
            pd.DataFrame: ヒートマップ用データ
            
        Raises:
            ValueError: データがマージされていない場合
        """
        if self.merged_df is None:
            raise ValueError("Data must be merged before creating heatmap")
        
        # 路線×天候の遅延率マトリックス
        heatmap_data = self.merged_df.pivot_table(
            values='delay_minutes',
            index='line',
            columns='weather_main',
            aggfunc='mean',
            fill_value=0
        ).round(1)
        
        return heatmap_data
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """
        分析結果のサマリーレポートを生成
        
        Returns:
            Dict[str, Any]: サマリーレポート
        """
        if self.merged_df is None:
            return {"error": "No data available for analysis"}
        
        stats = self.calculate_delay_statistics()
        correlations = self.calculate_correlation()
        
        # 天候の日本語マッピング
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
        
        # 最も影響が大きい天候を日本語で取得
        most_impactful_weather = max(stats['avg_delay_by_weather'], key=stats['avg_delay_by_weather'].get)
        most_impactful_weather_jp = weather_jp_map.get(most_impactful_weather, most_impactful_weather)
        
        report = {
            "データ概要": {
                "分析期間": {
                    "開始": self.merged_df['timestamp'].min().strftime('%Y-%m-%d'),
                    "終了": self.merged_df['timestamp'].max().strftime('%Y-%m-%d')
                },
                "総レコード数": len(self.merged_df),
                "路線数": self.merged_df['line'].nunique(),
                "天候パターン数": self.merged_df['weather_main'].nunique()
            },
            "主要な発見": {
                "最も遅延が多い路線": max(stats['avg_delay_by_line'], key=stats['avg_delay_by_line'].get),
                "最も影響が大きい天候": most_impactful_weather_jp,
                "雨天時の遅延増加率": f"{correlations.get('rainy_vs_clear_ratio', 0):.1f}倍",
                "降雨量との相関": f"{correlations.get('rain_delay_corr', 0):.3f}"
            },
            "推奨事項": self._generate_recommendations(stats, correlations)
        }
        
        return report
    
    def _generate_recommendations(self, 
                                stats: Dict[str, Any], 
                                correlations: Dict[str, float]) -> List[str]:
        """
        分析結果に基づく推奨事項を生成
        
        Args:
            stats: 統計情報
            correlations: 相関情報
            
        Returns:
            List[str]: 推奨事項のリスト
        """
        recommendations = []
        
        # 雨天時の対策
        if correlations.get('rainy_vs_clear_ratio', 0) > 1.5:
            recommendations.append("雨天時は通常より早めの出発を推奨（特に梅雨時期）")
        
        # ラッシュアワー対策
        rush_stats = stats.get('rush_hour_analysis', {})
        if rush_stats.get('mean', {}).get(True, 0) > rush_stats.get('mean', {}).get(False, 0) * 1.5:
            recommendations.append("ラッシュアワーを避けた時差通勤の検討")
        
        # 特定路線の対策
        high_delay_lines = [
            line for line, delay in stats['avg_delay_by_line'].items() 
            if delay > 10
        ]
        if high_delay_lines:
            recommendations.append(f"特に遅延が多い路線（{', '.join(high_delay_lines)}）の代替ルート検討")
        
        return recommendations
    
    def generate_sample_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        テスト用のサンプルデータを生成
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 遅延データと天気データ
        """
        lines = Config.TRAIN_LINES
        weather_conditions = Config.WEATHER_CONDITIONS[:4]  # 主要な天候のみ
        
        # 30日分のデータを生成
        delay_records = []
        weather_records = []
        
        base_date = datetime.now() - timedelta(days=30)
        
        for day in range(30):
            current_date = base_date + timedelta(days=day)
            
            # 1日の天気を決定（実際の天候パターンに近づける）
            if day % 7 in [0, 6]:  # 週末は晴れやすい
                weather_probs = [0.5, 0.2, 0.2, 0.1]
            else:  # 平日
                weather_probs = [0.3, 0.3, 0.3, 0.1]
            
            daily_weather = np.random.choice(weather_conditions, p=weather_probs)
            is_rainy = daily_weather in ['Rain', 'Drizzle']
            
            # 天気データ生成
            weather_records.extend(
                self._generate_daily_weather_records(current_date, daily_weather, is_rainy)
            )
            
            # 遅延データ生成
            delay_records.extend(
                self._generate_daily_delay_records(current_date, is_rainy, lines)
            )
        
        # DataFrameに変換
        delay_df = pd.DataFrame(delay_records)
        weather_df = pd.DataFrame(weather_records)
        
        # 保存
        os.makedirs(Config.RAW_DATA_DIR, exist_ok=True)
        delay_df.to_csv(
            os.path.join(Config.RAW_DATA_DIR, 'sample_train_delays.csv'), 
            index=False, 
            encoding='utf-8-sig'
        )
        weather_df.to_csv(
            os.path.join(Config.RAW_DATA_DIR, 'sample_weather_data.csv'), 
            index=False, 
            encoding='utf-8-sig'
        )
        
        self.logger.info(
            f"Generated {len(delay_df)} delay records and "
            f"{len(weather_df)} weather records"
        )
        
        return delay_df, weather_df
    
    def _generate_daily_weather_records(self, 
                                      date: datetime, 
                                      weather: str, 
                                      is_rainy: bool) -> List[Dict[str, Any]]:
        """
        1日分の天気レコードを生成
        
        Args:
            date: 日付
            weather: 天候
            is_rainy: 雨天かどうか
            
        Returns:
            List[Dict[str, Any]]: 天気レコードのリスト
        """
        records = []
        
        for hour in range(24):
            weather_time = date + timedelta(hours=hour)
            
            # 時間による温度変化
            base_temp = 20
            temp_variation = -5 * np.cos(2 * np.pi * (hour - 14) / 24)
            temperature = base_temp + temp_variation + np.random.normal(0, 2)
            
            records.append({
                'timestamp': weather_time,
                'weather_main': weather,
                'temperature': round(temperature, 1),
                'humidity': round(np.random.normal(70 if is_rainy else 50, 10), 1),
                'pressure': round(np.random.normal(1013, 5), 1),
                'wind_speed': round(np.random.exponential(3), 1),
                'rain_1h': round(np.random.exponential(2) if is_rainy else 0, 1),
                'is_rainy': is_rainy
            })
        
        return records
    
    def _generate_daily_delay_records(self, 
                                    date: datetime, 
                                    is_rainy: bool, 
                                    lines: List[str]) -> List[Dict[str, Any]]:
        """
        1日分の遅延レコードを生成
        
        Args:
            date: 日付
            is_rainy: 雨天かどうか
            lines: 路線リスト
            
        Returns:
            List[Dict[str, Any]]: 遅延レコードのリスト
        """
        records = []
        
        for hour in range(24):
            delay_time = date + timedelta(hours=hour)
            
            # ラッシュ時の判定
            is_rush_hour = any(start <= hour <= end for start, end in Config.RUSH_HOURS)
            
            for line in lines:
                # 基本遅延確率（路線特性を考慮）
                if line in ['山手線', '中央線']:  # 主要路線は遅延しやすい
                    base_delay_prob = 0.4 if is_rush_hour else 0.15
                else:
                    base_delay_prob = 0.3 if is_rush_hour else 0.1
                
                # 雨天時は遅延確率増加
                if is_rainy:
                    base_delay_prob *= 1.5
                
                # 遅延するかどうか
                if np.random.random() < base_delay_prob:
                    # 遅延時間を決定
                    if is_rainy and is_rush_hour:
                        delay_minutes = np.random.gamma(2, 5)  # より長い遅延
                    elif is_rainy or is_rush_hour:
                        delay_minutes = np.random.exponential(8)
                    else:
                        delay_minutes = np.random.exponential(5)
                    
                    reason = self._determine_delay_reason(is_rainy, is_rush_hour)
                else:
                    delay_minutes = 0
                    reason = None
                
                records.append({
                    'timestamp': delay_time,
                    'line': line,
                    'delay_minutes': min(delay_minutes, 60),  # 最大60分
                    'reason': reason,
                    'status': 'delayed' if delay_minutes > 0 else 'normal'
                })
        
        return records
    
    def _determine_delay_reason(self, is_rainy: bool, is_rush_hour: bool) -> str:
        """
        遅延理由を決定
        
        Args:
            is_rainy: 雨天かどうか
            is_rush_hour: ラッシュアワーかどうか
            
        Returns:
            str: 遅延理由
        """
        if is_rainy and is_rush_hour:
            reasons = ['天候', '混雑', '天候による混雑']
            probs = [0.3, 0.3, 0.4]
        elif is_rainy:
            reasons = ['天候', '信号確認', '車両点検']
            probs = [0.6, 0.2, 0.2]
        elif is_rush_hour:
            reasons = ['混雑', '車両点検', '信号確認']
            probs = [0.7, 0.2, 0.1]
        else:
            reasons = ['車両点検', '信号確認', 'その他']
            probs = [0.4, 0.3, 0.3]
        
        return np.random.choice(reasons, p=probs)


def main():
    """
    メイン実行関数
    """
    # 設定検証
    Config.validate()
    
    # アナライザー初期化
    analyzer = TrainDelayAnalyzer()
    
    # サンプルデータを生成
    analyzer.generate_sample_data()
    
    # データを読み込んで分析
    analyzer.load_data(
        os.path.join(Config.RAW_DATA_DIR, 'sample_train_delays.csv'),
        os.path.join(Config.RAW_DATA_DIR, 'sample_weather_data.csv')
    )
    analyzer.merge_data()
    
    # 統計を計算
    stats = analyzer.calculate_delay_statistics()
    correlations = analyzer.calculate_correlation()
    report = analyzer.generate_summary_report()
    
    # 結果を表示
    print("\n" + "="*50)
    print("電車遅延×天気データ 相関分析レポート")
    print("="*50)
    
    print("\n【データ概要】")
    for key, value in report['データ概要'].items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  - {k}: {v}")
        else:
            print(f"{key}: {value}")
    
    print("\n【主要な発見】")
    for key, value in report['主要な発見'].items():
        print(f"- {key}: {value}")
    
    print("\n【推奨事項】")
    for i, rec in enumerate(report['推奨事項'], 1):
        print(f"{i}. {rec}")
    
    print("\n【詳細統計】")
    print(f"路線別平均遅延時間: {stats['avg_delay_by_line']}")
    print(f"天候別平均遅延時間: {stats['avg_delay_by_weather']}")
    
    print("\n【相関分析】")
    print(f"降雨量と遅延の相関係数: {correlations.get('rain_delay_corr', 0):.3f}")
    print(f"湿度と遅延の相関係数: {correlations.get('humidity_delay_corr', 0):.3f}")
    print(f"気温と遅延の相関係数: {correlations.get('temp_delay_corr', 0):.3f}")


if __name__ == "__main__":
    main()