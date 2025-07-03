# 開発者ガイド

## 目次

1. [開発環境のセットアップ](#開発環境のセットアップ)
2. [プロジェクト構造](#プロジェクト構造)
3. [コーディング規約](#コーディング規約)
4. [新機能の追加方法](#新機能の追加方法)
5. [テストの実行](#テストの実行)
6. [デバッグ方法](#デバッグ方法)
7. [コントリビューション](#コントリビューション)

## 開発環境のセットアップ

### 必要なツール

- Python 3.8+
- Git
- VSCode または PyCharm（推奨）

### 環境構築手順

```bash
# リポジトリのクローン
git clone <repository-url>
cd data-visualization

# 仮想環境の作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 開発用依存関係のインストール
pip install -r requirements.txt
pip install -r requirements-dev.txt  # 開発ツール
```

### VSCode推奨拡張機能

`.vscode/extensions.json`:
```json
{
    "recommendations": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "ms-python.black-formatter",
        "charliermarsh.ruff",
        "njpwerner.autodocstring"
    ]
}
```

## プロジェクト構造

```
data-visualization/
├── app.py                  # メインアプリケーション
├── src/                    # ソースコード
│   ├── __init__.py
│   ├── base_collector.py   # 基底クラス
│   ├── config.py          # 設定管理
│   ├── data_analyzer.py   # 分析ロジック
│   ├── train_delay_collector.py  # 遅延データ収集
│   ├── weather_collector.py      # 天気データ収集
│   └── visualization.py          # 可視化
├── data/                  # データディレクトリ
│   ├── raw/              # 生データ
│   └── processed/        # 処理済みデータ
├── docs/                  # ドキュメント
├── tests/                 # テストコード
├── logs/                  # ログファイル
└── requirements.txt       # 依存関係
```

## コーディング規約

### Python スタイルガイド

PEP 8に準拠し、以下の追加規約を適用：

```python
# インポートの順序
import os
import sys
from datetime import datetime

import pandas as pd
import numpy as np

from src.config import Config
from src.base_collector import BaseCollector
```

### 命名規則

```python
# クラス名: PascalCase
class TrainDelayAnalyzer:
    pass

# 関数名・変数名: snake_case
def calculate_delay_statistics():
    delay_minutes = 10
    
# 定数: UPPER_SNAKE_CASE
MAX_DELAY_MINUTES = 60
DEFAULT_CITY = "Tokyo"

# プライベートメソッド: アンダースコアプレフィックス
def _validate_data():
    pass
```

### 型ヒント

すべての関数に型ヒントを追加：

```python
from typing import Dict, List, Optional, Any

def analyze_delays(
    delay_data: pd.DataFrame,
    weather_data: pd.DataFrame,
    threshold: float = 5.0
) -> Dict[str, Any]:
    """
    遅延データを分析
    
    Args:
        delay_data: 遅延データ
        weather_data: 天気データ
        threshold: 遅延判定の閾値（分）
        
    Returns:
        Dict[str, Any]: 分析結果
    """
    pass
```

### ドキュメント文字列

Google スタイルのdocstringを使用：

```python
def merge_data(self, delay_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    遅延データと天気データをマージ
    
    Args:
        delay_df: 遅延データのDataFrame
        weather_df: 天気データのDataFrame
        
    Returns:
        pd.DataFrame: マージされたデータ
        
    Raises:
        ValueError: データ形式が不正な場合
        
    Example:
        >>> analyzer = DataAnalyzer()
        >>> merged = analyzer.merge_data(delays, weather)
    """
    pass
```

## 新機能の追加方法

### 1. 新しいデータソースの追加

```python
# src/new_data_collector.py
from src.base_collector import BaseCollector

class NewDataCollector(BaseCollector):
    """新しいデータソースからデータを収集"""
    
    def __init__(self):
        super().__init__(logger_name=__name__)
        
    def collect_data(self) -> pd.DataFrame:
        """データ収集の実装"""
        # 実装
        pass
```

### 2. 新しい可視化の追加

```python
# src/visualization.py に追加
def create_new_visualization(self, data: pd.DataFrame) -> go.Figure:
    """新しい可視化を作成"""
    
    fig = go.Figure()
    
    # グラフの作成
    fig.add_trace(go.Scatter(
        x=data['x'],
        y=data['y'],
        mode='lines+markers',
        name='データ'
    ))
    
    # レイアウトの設定
    fig.update_layout(
        title="新しい可視化",
        xaxis_title="X軸",
        yaxis_title="Y軸"
    )
    
    return fig
```

### 3. 新しい分析手法の追加

```python
# src/data_analyzer.py に追加
def calculate_new_metric(self) -> Dict[str, float]:
    """新しい指標を計算"""
    
    if self.merged_df is None:
        raise ValueError("Data must be merged first")
    
    # 計算ロジック
    result = {
        'metric1': self._calculate_metric1(),
        'metric2': self._calculate_metric2()
    }
    
    return result
```

## テストの実行

### ユニットテストの作成

```python
# tests/test_data_analyzer.py
import unittest
import pandas as pd
from src.data_analyzer import TrainDelayAnalyzer

class TestTrainDelayAnalyzer(unittest.TestCase):
    
    def setUp(self):
        """テストの前準備"""
        self.analyzer = TrainDelayAnalyzer()
        
    def test_load_data(self):
        """データ読み込みのテスト"""
        # テストデータの準備
        delay_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10),
            'line': ['山手線'] * 10,
            'delay_minutes': [0, 5, 10, 0, 15, 0, 0, 20, 0, 5]
        })
        
        # テストの実行
        self.analyzer.delay_df = delay_data
        
        # 検証
        self.assertEqual(len(self.analyzer.delay_df), 10)
        self.assertIn('timestamp', self.analyzer.delay_df.columns)
```

### テストの実行

```bash
# 全テストの実行
python -m pytest tests/

# 特定のテストファイルの実行
python -m pytest tests/test_data_analyzer.py

# カバレッジレポート付き
python -m pytest --cov=src tests/
```

## デバッグ方法

### ログの活用

```python
import logging

logger = logging.getLogger(__name__)

def process_data(data: pd.DataFrame) -> pd.DataFrame:
    logger.debug(f"Processing {len(data)} records")
    
    try:
        # 処理
        result = data.copy()
        logger.info("Data processing completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error processing data: {e}", exc_info=True)
        raise
```

### Streamlitのデバッグ

```python
# デバッグ情報の表示
if st.checkbox("デバッグ情報を表示"):
    st.write("Current data shape:", filtered_df.shape)
    st.write("Memory usage:", filtered_df.memory_usage().sum() / 1024**2, "MB")
    st.write("Column types:", filtered_df.dtypes)
```

### ブレークポイントの使用

```python
# VSCodeでのデバッグ
import pdb

def complex_calculation(data):
    # ブレークポイント
    pdb.set_trace()
    
    result = data * 2
    return result
```

## パフォーマンス最適化

### プロファイリング

```python
import cProfile
import pstats

def profile_function():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # 計測したい処理
    analyzer = TrainDelayAnalyzer()
    analyzer.generate_sample_data()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
```

### メモリ使用量の最適化

```python
# 大きなDataFrameの処理
def process_large_data(file_path: str, chunksize: int = 10000):
    """大きなファイルをチャンクで処理"""
    
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        # チャンクごとの処理
        processed = process_chunk(chunk)
        chunks.append(processed)
    
    return pd.concat(chunks, ignore_index=True)
```

## デプロイメント

### ローカルビルド

```bash
# Dockerイメージのビルド
docker build -t train-delay-analysis .

# コンテナの実行
docker run -p 8501:8501 train-delay-analysis
```

### 本番環境へのデプロイ

```bash
# Streamlit Cloudへのデプロイ
# 1. GitHubにプッシュ
git push origin main

# 2. Streamlit Cloudで設定
# - リポジトリを接続
# - app.pyを指定
# - 環境変数を設定
```

## コントリビューション

### ブランチ戦略

```bash
# 機能開発
git checkout -b feature/new-visualization

# バグ修正
git checkout -b bugfix/data-loading-error

# ドキュメント更新
git checkout -b docs/update-user-guide
```

### コミットメッセージ

```bash
# 良い例
git commit -m "feat: 降雪データの収集機能を追加"
git commit -m "fix: 日付フィルターのバグを修正"
git commit -m "docs: APIドキュメントを更新"
git commit -m "refactor: データ分析クラスをリファクタリング"

# 避けるべき例
git commit -m "更新"
git commit -m "バグ修正"
```

### プルリクエスト

1. フォークしてローカルで開発
2. テストを追加・実行
3. ドキュメントを更新
4. プルリクエストを作成

**PRテンプレート:**
```markdown
## 概要
変更の概要を記載

## 変更内容
- [ ] 新機能の追加
- [ ] バグ修正
- [ ] リファクタリング
- [ ] ドキュメント更新

## テスト
- [ ] ユニットテストを追加
- [ ] 既存のテストが通ることを確認
- [ ] 手動テストを実施

## スクリーンショット
（UIの変更がある場合）
```

## トラブルシューティング

### よくある開発時の問題

#### インポートエラー

```python
# 問題
ModuleNotFoundError: No module named 'src'

# 解決方法
# プロジェクトルートから実行
python app.py

# またはPYTHONPATHを設定
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### メモリ不足

```python
# 問題
MemoryError during data processing

# 解決方法
# データ型の最適化
df['delay_minutes'] = df['delay_minutes'].astype('int16')
df['line'] = df['line'].astype('category')
```

## リソース

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Python Documentation](https://plotly.com/python/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)