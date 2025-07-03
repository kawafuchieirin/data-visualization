"""
電車遅延データ収集モジュール
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import json
import logging
from typing import Dict, List, Optional, Any
import time
import os
import sys

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.base_collector import BaseCollector
from src.config import Config

logging.basicConfig(level=logging.INFO, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class TrainDelayCollector(BaseCollector):
    """
    電車遅延データを収集するクラス
    
    Attributes:
        railway_companies: 鉄道会社ごとの収集メソッドマッピング
        delay_reasons: 遅延理由のカテゴリ
    """
    
    def __init__(self):
        """
        コレクターを初期化
        """
        super().__init__(logger_name=__name__)
        self.delay_data: List[Dict[str, Any]] = []
        self.railway_companies = {
            'jr_east': self.scrape_jr_east_delays,
            'metro': self.scrape_metro_delays
        }
        self.delay_reasons = ['車両点検', '混雑', '信号確認', '天候', '人身事故', 'その他']
        
    def scrape_jr_east_delays(self) -> List[Dict[str, Any]]:
        """
        JR東日本の運行情報を取得（サンプル実装）
        
        Note:
            実際の実装では各鉄道会社のAPIや公式サイトに合わせて調整が必要
            
        Returns:
            List[Dict[str, Any]]: 遅延情報のリスト
        """
        try:
            # サンプルデータ生成（実際にはAPIやスクレイピング）
            sample_delays = self._generate_sample_delays(
                lines=['山手線', '中央線', '京浜東北線', '東海道線'],
                company='JR東日本'
            )
            return sample_delays
        except Exception as e:
            self.logger.error(f"Error scraping JR East delays: {e}")
            return []
    
    def scrape_metro_delays(self) -> List[Dict[str, Any]]:
        """
        東京メトロの運行情報を取得（サンプル実装）
        
        Returns:
            List[Dict[str, Any]]: 遅延情報のリスト
        """
        try:
            # サンプルデータ生成
            sample_delays = self._generate_sample_delays(
                lines=['銀座線', '丸ノ内線', '千代田線'],
                company='東京メトロ'
            )
            return sample_delays
        except Exception as e:
            self.logger.error(f"Error scraping Metro delays: {e}")
            return []
    
    def collect_data(self) -> pd.DataFrame:
        """
        全路線の遅延情報を収集
        
        Returns:
            pd.DataFrame: 収集した遅延データ
        """
        all_delays = []
        
        # 各鉄道会社のデータを収集
        for company_name, scrape_method in self.railway_companies.items():
            try:
                delays = scrape_method()
                all_delays.extend(delays)
                self.logger.info(f"Collected {len(delays)} {company_name} delay records")
            except Exception as e:
                self.logger.error(f"Error collecting {company_name} delays: {e}")
        
        # DataFrameに変換
        df = pd.DataFrame(all_delays)
        
        # データ検証
        required_columns = ['line', 'delay_minutes', 'timestamp', 'status']
        if self.validate_data(df, required_columns):
            return df
        else:
            self.logger.warning("Data validation failed, returning empty DataFrame")
            return pd.DataFrame()
    
    def _generate_sample_delays(self, lines: List[str], company: str) -> List[Dict[str, Any]]:
        """
        サンプル遅延データを生成（テスト用）
        
        Args:
            lines: 路線名のリスト
            company: 鉄道会社名
            
        Returns:
            List[Dict[str, Any]]: サンプル遅延データ
        """
        import random
        
        delays = []
        current_time = datetime.now()
        
        for line in lines:
            # 遅延確率（時間帯により変動）
            hour = current_time.hour
            is_rush_hour = any(start <= hour <= end for start, end in Config.RUSH_HOURS)
            delay_prob = 0.4 if is_rush_hour else 0.1
            
            if random.random() < delay_prob:
                delay_minutes = random.randint(5, 30)
                reason = random.choice(self.delay_reasons)
                status = 'delayed'
            else:
                delay_minutes = 0
                reason = None
                status = 'normal'
            
            delays.append({
                'line': line,
                'company': company,
                'delay_minutes': delay_minutes,
                'reason': reason,
                'timestamp': current_time.isoformat(),
                'status': status,
                'last_update': current_time.isoformat()
            })
        
        return delays

def main():
    """
    メイン実行関数
    """
    # 設定検証
    Config.validate()
    
    # コレクター初期化
    collector = TrainDelayCollector()
    
    # データ収集
    delays_df = collector.collect_data()
    
    if not delays_df.empty:
        # ファイル名生成
        filename = collector.get_timestamp_filename('train_delays')
        filepath = os.path.join(Config.RAW_DATA_DIR, filename)
        
        # データ保存
        collector.save_data(delays_df, filepath)
    else:
        logger.error("No delay data collected")

if __name__ == "__main__":
    main()