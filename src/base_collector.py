"""
基底コレクタークラス - データ収集の共通機能を提供
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import logging
import pandas as pd
from datetime import datetime

class BaseCollector(ABC):
    """データ収集の基底クラス"""
    
    def __init__(self, logger_name: str):
        """
        コレクターを初期化
        
        Args:
            logger_name: ロガー名
        """
        self.logger = logging.getLogger(logger_name)
        self.data: List[Dict[str, Any]] = []
        
    @abstractmethod
    def collect_data(self) -> pd.DataFrame:
        """
        データを収集する抽象メソッド
        
        Returns:
            pd.DataFrame: 収集したデータ
        """
        pass
    
    def save_data(self, df: pd.DataFrame, filepath: str, encoding: str = 'utf-8-sig') -> None:
        """
        データをCSVファイルに保存
        
        Args:
            df: 保存するDataFrame
            filepath: 保存先のファイルパス
            encoding: エンコーディング（デフォルト: utf-8-sig）
        """
        try:
            df.to_csv(filepath, index=False, encoding=encoding)
            self.logger.info(f"Saved {len(df)} records to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving data to {filepath}: {e}")
            raise
    
    def validate_data(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        データの妥当性を検証
        
        Args:
            df: 検証するDataFrame
            required_columns: 必須カラムのリスト
            
        Returns:
            bool: 検証結果
        """
        if df.empty:
            self.logger.warning("DataFrame is empty")
            return False
            
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False
            
        return True
    
    @staticmethod
    def get_timestamp_filename(prefix: str, extension: str = 'csv') -> str:
        """
        タイムスタンプ付きのファイル名を生成
        
        Args:
            prefix: ファイル名のプレフィックス
            extension: ファイル拡張子
            
        Returns:
            str: タイムスタンプ付きファイル名
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}.{extension}"