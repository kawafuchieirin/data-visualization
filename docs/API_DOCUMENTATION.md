# API ドキュメント

## 概要

このドキュメントでは、電車遅延×天気データ相関分析システムの各モジュールのAPIリファレンスを提供します。

## モジュール構成

### 1. データ収集モジュール

#### `src.train_delay_collector`

電車の遅延情報を収集するモジュールです。

```python
class TrainDelayCollector(BaseCollector):
    """
    電車遅延データを収集するクラス
    """
    
    def __init__(self):
        """コレクターを初期化"""
        
    def collect_data(self) -> pd.DataFrame:
        """
        全路線の遅延情報を収集
        
        Returns:
            pd.DataFrame: 収集した遅延データ
        """
        
    def scrape_jr_east_delays(self) -> List[Dict[str, Any]]:
        """
        JR東日本の運行情報を取得
        
        Returns:
            List[Dict[str, Any]]: 遅延情報のリスト
        """
        
    def scrape_metro_delays(self) -> List[Dict[str, Any]]:
        """
        東京メトロの運行情報を取得
        
        Returns:
            List[Dict[str, Any]]: 遅延情報のリスト
        """
```

**出力データ形式:**
```json
{
    "line": "山手線",
    "company": "JR東日本",
    "delay_minutes": 15,
    "reason": "車両点検",
    "timestamp": "2024-01-01T10:00:00",
    "status": "delayed",
    "last_update": "2024-01-01T10:00:00"
}
```

#### `src.weather_collector`

天気データを収集するモジュールです。

```python
class WeatherCollector(BaseCollector):
    """
    天気データを収集するクラス
    """
    
    def __init__(self):
        """コレクターを初期化"""
        
    def get_weather_data(self, 
                        city: str = None, 
                        lat: float = None, 
                        lon: float = None) -> Dict[str, Any]:
        """
        指定された都市の現在の天気データを取得
        
        Args:
            city: 都市名
            lat: 緯度
            lon: 経度
            
        Returns:
            Dict[str, Any]: 天気データ
        """
        
    def collect_historical_weather(self, days: int = 30) -> pd.DataFrame:
        """
        過去の天気データを収集
        
        Args:
            days: 収集する日数
            
        Returns:
            pd.DataFrame: 過去の天気データ
        """
```

**出力データ形式:**
```json
{
    "timestamp": "2024-01-01T10:00:00",
    "city": "Tokyo",
    "weather_main": "Rain",
    "weather_description": "小雨",
    "temperature": 15.5,
    "humidity": 85,
    "pressure": 1013,
    "wind_speed": 3.5,
    "rain_1h": 2.5,
    "is_rainy": true
}
```

### 2. データ分析モジュール

#### `src.data_analyzer`

遅延と天気の相関を分析するモジュールです。

```python
class TrainDelayAnalyzer:
    """
    電車遅延と天気データの相関を分析するクラス
    """
    
    def load_data(self, delay_file: str, weather_file: str) -> None:
        """
        遅延データと天気データを読み込み
        
        Args:
            delay_file: 遅延データファイルのパス
            weather_file: 天気データファイルのパス
        """
        
    def merge_data(self) -> pd.DataFrame:
        """
        遅延データと天気データを時間でマージ
        
        Returns:
            pd.DataFrame: マージされたデータ
        """
        
    def calculate_delay_statistics(self) -> Dict[str, Any]:
        """
        遅延統計を計算
        
        Returns:
            Dict[str, Any]: 各種統計情報
        """
        
    def calculate_correlation(self) -> Dict[str, float]:
        """
        天候要因と遅延の相関を計算
        
        Returns:
            Dict[str, float]: 各種相関係数
        """
        
    def create_heatmap_data(self) -> pd.DataFrame:
        """
        ヒートマップ用のデータを作成
        
        Returns:
            pd.DataFrame: ヒートマップ用データ
        """
        
    def generate_summary_report(self) -> Dict[str, Any]:
        """
        分析結果のサマリーレポートを生成
        
        Returns:
            Dict[str, Any]: サマリーレポート
        """
```

**統計情報の形式:**
```python
{
    'avg_delay_by_line': {'山手線': 5.2, '中央線': 6.8, ...},
    'avg_delay_by_weather': {'Clear': 2.1, 'Rain': 8.5, ...},
    'delay_rate_by_line_weather': DataFrame,
    'avg_delay_by_hour': {0: 1.2, 1: 0.8, ..., 23: 2.1},
    'avg_delay_by_day': {'Monday': 5.2, 'Tuesday': 4.8, ...},
    'rush_hour_analysis': {'mean': {True: 8.5, False: 2.1}, ...}
}
```

### 3. 可視化モジュール

#### `src.visualization`

データの可視化を担当するモジュールです。

```python
class Visualizer:
    """
    データ可視化を担当するクラス
    """
    
    def create_delay_by_weather_bar(self, data: pd.DataFrame) -> go.Figure:
        """天候別平均遅延時間の棒グラフを作成"""
        
    def create_delay_by_line_bar(self, data: pd.DataFrame) -> go.Figure:
        """路線別平均遅延時間の横棒グラフを作成"""
        
    def create_delay_heatmap(self, heatmap_data: pd.DataFrame) -> go.Figure:
        """路線×天候の遅延ヒートマップを作成"""
        
    def create_daily_trend_line(self, data: pd.DataFrame) -> go.Figure:
        """日別遅延推移の折れ線グラフを作成"""
        
    def create_hourly_pattern_bar(self, data: pd.DataFrame) -> go.Figure:
        """時間帯別遅延パターンの棒グラフを作成"""
        
    def create_weekday_pattern_bar(self, data: pd.DataFrame) -> go.Figure:
        """曜日別遅延パターンの棒グラフを作成"""
        
    def create_correlation_scatter(self, data: pd.DataFrame, 
                                 x_col: str, y_col: str) -> go.Figure:
        """相関散布図を作成"""
```

### 4. 設定モジュール

#### `src.config`

アプリケーション全体の設定を管理するモジュールです。

```python
class Config:
    """アプリケーション設定クラス"""
    
    # API設定
    WEATHER_API_KEY: str
    WEATHER_API_BASE_URL: str
    
    # データ収集設定
    TRAIN_LINES: List[str]
    WEATHER_CONDITIONS: List[str]
    
    # 地域設定
    DEFAULT_CITY: str
    DEFAULT_LAT: float
    DEFAULT_LON: float
    
    # データパス
    DATA_DIR: str
    RAW_DATA_DIR: str
    PROCESSED_DATA_DIR: str
    
    # 分析設定
    RUSH_HOURS: List[tuple]
    
    # 可視化設定
    COLOR_SCHEMES: Dict[str, str]
    
    @classmethod
    def validate(cls) -> bool:
        """設定の妥当性を検証"""
```

## 使用例

### 1. データ収集

```python
from src.train_delay_collector import TrainDelayCollector
from src.weather_collector import WeatherCollector

# 遅延データの収集
delay_collector = TrainDelayCollector()
delay_df = delay_collector.collect_data()

# 天気データの収集
weather_collector = WeatherCollector()
weather_df = weather_collector.collect_historical_weather(days=30)
```

### 2. データ分析

```python
from src.data_analyzer import TrainDelayAnalyzer

# 分析の実行
analyzer = TrainDelayAnalyzer()
analyzer.load_data('delay_data.csv', 'weather_data.csv')
merged_df = analyzer.merge_data()

# 統計の計算
stats = analyzer.calculate_delay_statistics()
correlations = analyzer.calculate_correlation()
report = analyzer.generate_summary_report()
```

### 3. 可視化

```python
from src.visualization import Visualizer

# グラフの作成
visualizer = Visualizer()
fig = visualizer.create_delay_heatmap(heatmap_data)
fig.show()
```

## エラーハンドリング

各モジュールは以下の例外を発生させる可能性があります：

- `FileNotFoundError`: ファイルが見つからない場合
- `ValueError`: データ形式が不正な場合
- `ConnectionError`: API接続エラー
- `KeyError`: 必須フィールドが欠落している場合

## 拡張ポイント

### 新しい鉄道会社の追加

`TrainDelayCollector`クラスに新しいスクレイピングメソッドを追加：

```python
def scrape_custom_railway(self) -> List[Dict[str, Any]]:
    """カスタム鉄道会社の遅延情報を取得"""
    # 実装
    pass

# railway_companiesディクショナリに追加
self.railway_companies['custom'] = self.scrape_custom_railway
```

### 新しい可視化の追加

`Visualizer`クラスに新しいメソッドを追加：

```python
def create_custom_visualization(self, data: pd.DataFrame) -> go.Figure:
    """カスタム可視化を作成"""
    # 実装
    pass
```