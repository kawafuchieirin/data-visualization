"""
設定ファイル - アプリケーション全体の設定を管理
"""
from typing import List, Dict
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """アプリケーション設定クラス"""
    
    # API設定
    WEATHER_API_KEY: str = os.getenv('WEATHER_API_KEY', '')
    WEATHER_API_BASE_URL: str = "http://api.openweathermap.org/data/2.5/weather"
    
    # データ収集設定
    TRAIN_LINES: List[str] = ['山手線']
    
    WEATHER_CONDITIONS: List[str] = ['Clear', 'Rain', 'Clouds', 'Drizzle', 'Thunderstorm', 'Snow']
    
    # 地域設定
    DEFAULT_CITY: str = "Tokyo"
    DEFAULT_LAT: float = 35.6762
    DEFAULT_LON: float = 139.6503
    
    # データパス
    DATA_DIR: str = "data"
    RAW_DATA_DIR: str = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR: str = os.path.join(DATA_DIR, "processed")
    
    # ログ設定
    LOG_DIR: str = "logs"
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 分析設定
    RUSH_HOURS: List[tuple] = [(7, 9), (17, 20)]  # 朝と夕方のラッシュアワー
    
    # 可視化設定
    COLOR_SCHEMES: Dict[str, str] = {
        'heatmap': 'RdYlBu_r',
        'delay_bar': 'Reds',
        'line_bar': 'Blues',
        'time_series': 'Viridis'
    }
    
    # Streamlit設定
    PAGE_TITLE: str = "山手線遅延×天気 相関分析ダッシュボード"
    PAGE_ICON: str = "🚃"
    LAYOUT: str = "wide"
    
    @classmethod
    def validate(cls) -> bool:
        """設定の妥当性を検証"""
        # 必要なディレクトリを作成
        os.makedirs(cls.RAW_DATA_DIR, exist_ok=True)
        os.makedirs(cls.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        
        return True