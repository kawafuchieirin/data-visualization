"""
天気データ収集モジュール
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
import os
import sys
from dotenv import load_dotenv
from typing import Dict, List, Optional, Any
import random
import numpy as np

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.base_collector import BaseCollector
from src.config import Config

load_dotenv()
logging.basicConfig(level=logging.INFO, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class WeatherCollector(BaseCollector):
    """
    天気データを収集するクラス
    
    Attributes:
        api_key: Weather APIのキー
        base_url: APIのベースURL
        weather_descriptions: 天気の日本語説明マッピング
    """
    
    def __init__(self):
        """
        コレクターを初期化
        """
        super().__init__(logger_name=__name__)
        self.api_key = os.getenv('WEATHER_API_KEY', Config.WEATHER_API_KEY)
        self.base_url = Config.WEATHER_API_BASE_URL
        self.weather_descriptions = {
            'Clear': '晴れ',
            'Rain': '雨',
            'Clouds': '曇り',
            'Drizzle': '霧雨',
            'Thunderstorm': '雷雨',
            'Snow': '雪',
            'Mist': '霧',
            'Fog': '濃霧'
        }
        
    def get_weather_data(self, 
                        city: str = None, 
                        lat: float = None, 
                        lon: float = None) -> Dict[str, Any]:
        """
        指定された都市の現在の天気データを取得
        
        Args:
            city: 都市名（デフォルト: Config.DEFAULT_CITY）
            lat: 緯度（デフォルト: Config.DEFAULT_LAT）
            lon: 経度（デフォルト: Config.DEFAULT_LON）
            
        Returns:
            Dict[str, Any]: 天気データ
        """
        # デフォルト値を設定
        city = city or Config.DEFAULT_CITY
        lat = lat or Config.DEFAULT_LAT
        lon = lon or Config.DEFAULT_LON
        if self.api_key:
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric',
                'lang': 'ja'
            }
            
            try:
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logger.error(f"Error fetching weather data: {e}")
                return self._get_sample_weather_data()
        else:
            logger.warning("No API key found, using sample data")
            return self._get_sample_weather_data()
    
    def _get_sample_weather_data(self) -> Dict[str, Any]:
        """
        サンプル天気データを生成
        
        Returns:
            Dict[str, Any]: サンプル天気データ
        """
        weather_main = random.choice(Config.WEATHER_CONDITIONS)
        is_rainy = weather_main in ['Rain', 'Drizzle', 'Thunderstorm']
        
        return {
            'weather': [{
                'main': weather_main,
                'description': self.weather_descriptions.get(weather_main, weather_main)
            }],
            'main': {
                'temp': random.uniform(10, 30),
                'humidity': random.uniform(70, 95) if is_rainy else random.uniform(40, 70),
                'pressure': random.uniform(1000, 1020)
            },
            'wind': {'speed': random.uniform(1, 10)},
            'rain': {'1h': random.uniform(0.5, 5) if is_rainy else 0},
            'dt': int(datetime.now().timestamp()),
            'name': Config.DEFAULT_CITY
        }
    
    def parse_weather_data(self, weather_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        天気データを解析用の形式に変換
        
        Args:
            weather_data: 生の天気データ
            
        Returns:
            Dict[str, Any]: 解析用に整形された天気データ
        """
        try:
            parsed = {
                'timestamp': datetime.fromtimestamp(weather_data['dt']).isoformat(),
                'city': weather_data.get('name', Config.DEFAULT_CITY),
                'weather_main': weather_data['weather'][0]['main'],
                'weather_description': weather_data['weather'][0]['description'],
                'temperature': round(weather_data['main']['temp'], 1),
                'humidity': weather_data['main']['humidity'],
                'pressure': weather_data['main']['pressure'],
                'wind_speed': round(weather_data['wind']['speed'], 1),
                'rain_1h': round(weather_data.get('rain', {}).get('1h', 0), 1),
                'is_rainy': weather_data['weather'][0]['main'] in ['Rain', 'Drizzle', 'Thunderstorm']
            }
            return parsed
        except KeyError as e:
            self.logger.error(f"Error parsing weather data: {e}")
            raise
    
    def collect_data(self) -> pd.DataFrame:
        """
        現在の天気データを収集
        
        Returns:
            pd.DataFrame: 収集した天気データ
        """
        weather_data = self.get_weather_data()
        parsed_data = self.parse_weather_data(weather_data)
        
        df = pd.DataFrame([parsed_data])
        
        # データ検証
        required_columns = ['timestamp', 'weather_main', 'temperature', 'humidity']
        if self.validate_data(df, required_columns):
            return df
        else:
            return pd.DataFrame()
    
    def collect_historical_weather(self, days: int = 30) -> pd.DataFrame:
        """
        過去の天気データを収集（サンプル実装）
        
        Args:
            days: 収集する日数
            
        Returns:
            pd.DataFrame: 過去の天気データ
            
        Note:
            実際の実装では履歴APIを使用
        """
        weather_records = []
        
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            
            # 1日の天気パターンを決定
            daily_pattern = self._generate_daily_weather_pattern(date)
            
            # 1時間ごとのデータを生成
            for hour in range(24):
                timestamp = date.replace(hour=hour, minute=0, second=0, microsecond=0)
                weather_data = self._generate_hourly_weather(timestamp, daily_pattern)
                weather_records.append(weather_data)
        
        return pd.DataFrame(weather_records)
    
    def _generate_daily_weather_pattern(self, date: datetime) -> Dict[str, Any]:
        """
        1日の天気パターンを生成
        
        Args:
            date: 対象日
            
        Returns:
            Dict[str, Any]: 天気パターン
        """
        # 季節性を考慮（簡易版）
        month = date.month
        if month in [6, 7, 8]:  # 夏：雨が多い
            weather_probs = [0.3, 0.4, 0.2, 0.1]  # Clear, Rain, Clouds, Drizzle
        elif month in [12, 1, 2]:  # 冬：晴れが多い
            weather_probs = [0.5, 0.1, 0.3, 0.1]
        else:  # 春秋
            weather_probs = [0.4, 0.2, 0.3, 0.1]
        
        weather_main = np.random.choice(
            ['Clear', 'Rain', 'Clouds', 'Drizzle'],
            p=weather_probs
        )
        
        # 基本温度（季節による）
        base_temps = {
            1: 5, 2: 7, 3: 12, 4: 18, 5: 22, 6: 25,
            7: 30, 8: 32, 9: 25, 10: 20, 11: 15, 12: 8
        }
        base_temp = base_temps.get(month, 20)
        
        return {
            'weather_main': weather_main,
            'base_temp': base_temp,
            'is_rainy': weather_main in ['Rain', 'Drizzle']
        }
    
    def _generate_hourly_weather(self, 
                               timestamp: datetime, 
                               daily_pattern: Dict[str, Any]) -> Dict[str, Any]:
        """
        時間ごとの天気データを生成
        
        Args:
            timestamp: タイムスタンプ
            daily_pattern: 1日の天気パターン
            
        Returns:
            Dict[str, Any]: 時間ごとの天気データ
        """
        hour = timestamp.hour
        
        # 温度の日変化（朝夕は低く、昼は高い）
        temp_variation = -5 * np.cos(2 * np.pi * (hour - 14) / 24)
        temperature = daily_pattern['base_temp'] + temp_variation + random.uniform(-2, 2)
        
        # 湿度（雨天時は高い）
        if daily_pattern['is_rainy']:
            humidity = random.uniform(70, 95)
        else:
            humidity = random.uniform(40, 70)
        
        # 降雨量（雨天時のみ）
        if daily_pattern['is_rainy']:
            # 時間帯による降雨量の変化
            rain_intensity = 1 + 0.5 * np.sin(2 * np.pi * hour / 24)
            rain_1h = max(0, rain_intensity * random.uniform(0.5, 3))
        else:
            rain_1h = 0
        
        return {
            'timestamp': timestamp.isoformat(),
            'city': Config.DEFAULT_CITY,
            'weather_main': daily_pattern['weather_main'],
            'weather_description': self.weather_descriptions.get(
                daily_pattern['weather_main'], 
                daily_pattern['weather_main']
            ),
            'temperature': round(temperature, 1),
            'humidity': round(humidity),
            'pressure': round(random.uniform(1000, 1020), 1),
            'wind_speed': round(random.uniform(1, 10), 1),
            'rain_1h': round(rain_1h, 1),
            'is_rainy': daily_pattern['is_rainy']
        }
    

def main():
    """
    メイン実行関数
    """
    # 設定検証
    Config.validate()
    
    # コレクター初期化
    collector = WeatherCollector()
    
    # 現在の天気を取得
    current_weather = collector.get_weather_data()
    parsed_weather = collector.parse_weather_data(current_weather)
    logger.info(f"Current weather: {parsed_weather}")
    
    # 履歴データを収集
    historical_df = collector.collect_historical_weather(days=30)
    
    if not historical_df.empty:
        # ファイル名生成
        filename = collector.get_timestamp_filename('weather_data')
        filepath = os.path.join(Config.RAW_DATA_DIR, filename)
        
        # データ保存
        collector.save_data(historical_df, filepath)
    else:
        logger.error("No weather data collected")

if __name__ == "__main__":
    main()