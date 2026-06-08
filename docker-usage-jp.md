# Vitoom Docker ユーザーガイド

[English](docker-usage-en.md) | [中文](docker-usage-cn.md) | **日本語**

## 1. 環境の準備

必要なもの：

- Docker
- Docker Compose

推論サービスを起動する場合はさらに：

- NVIDIA GPU
- **CUDA 13.0 対応の NVIDIA ドライバ**（`cu130` 推論イメージと一致）
- NVIDIA Container Toolkit

Docker から GPU にアクセスでき、CUDA 13.0 ランタイムが使えることを確認：

```bash
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
```

Windows では Docker Desktop + WSL2 を使用し、モデルディレクトリがあるディスクで **File Sharing** を有効にしてください。

## 2. 設定の準備

`.env` の生成は **どちらか一方のみ**（両方実行しない）：

### 方法 A：セットアップウィザード（推奨）

```bash
python scripts/setup_vitoom.py
```

コンポーネントを対話的に選択；アーキテクチャの自動検出、イメージ tag の書き込み、シークレット生成；末尾で Docker イメージ取得も可能。

### 方法 B：手動編集

```bash
cp .env.example .env
```

Windows PowerShell：`copy .env.example .env`

`.env` を編集し、デプロイに必要な項目をすべて記入（未設定だと起動失敗や誤ったイメージ取得の原因になります）：

```env
VITOOM_TARGET_ARCH=x86_64
VITOOM_INFERENCE_UPLOAD_AUTH_SECRET=長いランダム文字列を設定
VITOOM_SERVER_PORT=8888
VITOOM_BACKEND_URL=http://BACKEND_IP:8888
VITOOM_WS_URL=ws://BACKEND_IP:8888
```

イメージ tag（Docker Hub / オフライン tar と一致させること。aarch64 は `scripts/vitoom_setup/constants.py` を参照）：

```env
VITOOM_BACKEND_IMAGE=tonera/vitoom-backend:latest-x86_64
VITOOM_VISUAL_IMAGE=tonera/vitoom-inference-visual:experimental-cu130-torch2.11-x86_64
VITOOM_TEXT_IMAGE=tonera/vitoom-inference-text:experimental-cu130-torch2.11-x86_64
VITOOM_AUDIO_IMAGE=tonera/vitoom-inference-audio:experimental-cu130-torch2.9.1-x86_64
VITOOM_MINI_IMAGE=tonera/vitoom-inference-mini:experimental-cu130-torch2.11-x86_64
VITOOM_DOWNLOAD_IMAGE=tonera/vitoom-inference-download:experimental-x86_64
```

推論ノードでは Backend と同じシークレットと、そのホストにデプロイしたサービスの Supervisor URL を設定（未デプロイは空欄）：

```env
VITOOM_VISUAL_SUPERVISOR_URL=http://INFERENCE_IP:9001
VITOOM_TEXT_SUPERVISOR_URL=http://INFERENCE_IP:9002
VITOOM_AUDIO_SUPERVISOR_URL=http://INFERENCE_IP:9003
VITOOM_DOWNLOAD_SUPERVISOR_URL=http://INFERENCE_IP:9004
VITOOM_MINI_SUPERVISOR_URL=http://INFERENCE_IP:9005
```

Backend アドレスは LAN IP を使用し、`127.0.0.1` やコンテナ名は使わない。ポート変更時は `VITOOM_SERVER_PORT`、`VITOOM_BACKEND_URL`、`VITOOM_WS_URL` をまとめて更新。

---

`.env` の準備ができたら [§3 イメージの取得](#3-イメージの取得) へ（方法 A でウィザード内取得済みならスキップ可）。代表的なモデルの一括取得は [§6 初期モデルダウンロード（任意）](#6-初期モデルダウンロード任意) を参照。

## 3. イメージの取得

```bash
python scripts/load_vitoom_images.py
```

プロジェクト内 `images/<VITOOM_TARGET_ARCH>/` の tar を `docker load` で優先；なければ `docker pull`。

一部のサービスのみ：

```bash
python scripts/load_vitoom_images.py --components backend,visual
```

強制再取得：

```bash
python scripts/load_vitoom_images.py --force
```

## 4. Backend の起動

起動：

```bash
docker compose up -d backend
```

状態とログ：

```bash
docker compose ps
docker compose logs -f backend
```

ヘルスチェック：

```bash
curl http://127.0.0.1:8888/api/health
```

ブラウザ：

```text
http://127.0.0.1:8888
```

`VITOOM_SERVER_PORT` を変更した場合は、上記 URL のポートも合わせて変更。

## 5. 推論サービスの起動

Visual（画像・動画生成。初回は時間がかかります）：

```bash
docker compose -f docker-compose.inference.release.yml --profile visual up -d
```

Text（大規模言語モデル。初回は遅い。DGX Spark 上で約 5 分）：

```bash
docker compose -f docker-compose.inference.release.yml --profile text up -d
```

Audio（音声生成）：

```bash
docker compose -f docker-compose.inference.release.yml --profile audio up -d
```

Download（モデルダウンロードサービス）：

```bash
docker compose -f docker-compose.inference.release.yml --profile download up -d
```

Mini（小モデルサービス）：

```bash
docker compose -f docker-compose.inference.release.yml --profile mini up -d
```

状態確認：

```bash
docker compose -f docker-compose.inference.release.yml ps
```

ログ：

```bash
docker compose -f docker-compose.inference.release.yml logs -f visual
docker compose -f docker-compose.inference.release.yml logs -f text
docker compose -f docker-compose.inference.release.yml logs -f audio
docker compose -f docker-compose.inference.release.yml logs -f mini
docker compose -f docker-compose.inference.release.yml logs -f download
```

コンテナ内 supervisor の状態：

```bash
docker exec -it vitoom-inference-visual supervisorctl -s unix:///tmp/supervisor.sock status
docker exec -it vitoom-inference-text supervisorctl -s unix:///tmp/supervisor.sock status
docker exec -it vitoom-inference-audio supervisorctl -s unix:///tmp/supervisor.sock status
docker exec -it vitoom-inference-mini supervisorctl -s unix:///tmp/supervisor.sock status
```

## 6. 初期モデルダウンロード（任意）

`python scripts/setup_vitoom.py` で `.env` を生成した後、初期モデルスクリプトで体験を短縮できます。合計容量はおおよそ **100G+** です。

デプロイディレクトリ（`.env` があるリポジトリルート）で実行：

```bash
python scripts/download_initial_models.py
```

## 7. リソースディレクトリ

推論サービスのデフォルトマウント：

```text
resources/models
resources/weights
resources/loras
resources/outputs
```

別パスに置く場合は `.env` で指定：

```env
VITOOM_MODELS_HOST_DIR=/data/vitoom/models
VITOOM_WEIGHTS_HOST_DIR=/data/vitoom/weights
VITOOM_LORAS_HOST_DIR=/data/vitoom/loras
VITOOM_OUTPUTS_HOST_DIR=/data/vitoom/outputs
```

Windows ではスラッシュ区切り：

```env
VITOOM_MODELS_HOST_DIR=C:/vitoom/models
VITOOM_WEIGHTS_HOST_DIR=C:/vitoom/weights
VITOOM_LORAS_HOST_DIR=C:/vitoom/loras
VITOOM_OUTPUTS_HOST_DIR=C:/vitoom/outputs
```

## 8. データディレクトリ

Backend データはデプロイディレクトリの `data/` 配下：

```text
data/config             ユーザー設定
data/inference/config   推論サービス設定
data/resources          SQLite、出力、ナレッジベース、内蔵 ES データ
data/logs               Backend と ES のログ
data/logs/inference     推論サービスのログ
data/inference/cache    推論のコンパイル・加速キャッシュ
```

アップグレードや再起動時に `data/` は削除しないでください。

### Backend ログの確認

アプリログ：`data/logs/app.log` — `docker compose exec backend tail -f /app/logs/app.log`。内蔵 Elasticsearch：`data/logs/elasticsearch/`。

`.env` の `VITOOM_BACKEND_URL` / `VITOOM_WS_URL` 変更後は推論コンテナを再起動；entrypoint が `data/inference/config/inference.yaml` の `api_base_url` / `ws_url` を `.env` と同期します。

entrypoint テンプレートから設定ファイルを **丸ごと再生成**（`storage`、各サービス yaml など）する場合は一時的に：

```env
VITOOM_OVERWRITE_CONFIG=1
```

該当推論サービスを再起動：

```bash
docker compose -f docker-compose.inference.release.yml --profile visual up -d --force-recreate
```

反映確認後、`.env` を戻す：

```env
VITOOM_OVERWRITE_CONFIG=0
```

## 9. 分散デプロイ

Backend がコントロールプレーン。Visual、Text、Audio、Mini、Download はそれぞれ別 GPU サーバー上の推論ノードにできます。

推論マシンの `.env` では最低限 Backend に到達できること：

```env
VITOOM_BACKEND_URL=http://BACKEND_IP:8888
VITOOM_WS_URL=ws://BACKEND_IP:8888
VITOOM_INFERENCE_UPLOAD_AUTH_SECRET=Backend と同じシークレット
```

## 10. 停止とアップグレード

Backend 停止：

```bash
docker compose down
```

推論サービス停止：

```bash
docker compose -f docker-compose.inference.release.yml --profile visual down
docker compose -f docker-compose.inference.release.yml --profile text down
docker compose -f docker-compose.inference.release.yml --profile audio down
docker compose -f docker-compose.inference.release.yml --profile mini down
```

Backend アップグレード：

```bash
python scripts/load_vitoom_images.py --components backend --force
docker compose up -d backend
```

推論サービスのアップグレード：

```bash
python scripts/load_vitoom_images.py --components visual --force
docker compose -f docker-compose.inference.release.yml --profile visual up -d --force-recreate
```

アップグレード前のバックアップ推奨：

```text
data/
resources/
```

## 11. トラブルシューティング

Backend が正常か：

```bash
curl http://127.0.0.1:8888/api/health
docker compose logs --tail=200 backend
```

推論サービスが動いているか：

```bash
docker compose -f docker-compose.inference.release.yml ps
docker compose -f docker-compose.inference.release.yml logs --tail=200 visual
```

推論設定のアドレスが古くないか：

```bash
cat data/inference/config/inference.yaml
```

古い場合は `VITOOM_OVERWRITE_CONFIG=1` を設定して推論コンテナを再起動。

イメージ tag が `.env` と一致するか：

```bash
docker images | grep vitoom-inference
```

## 12. ヒント

以下は Docker デプロイを前提：Backend 設定は `data/config/`（初回起動でイメージから `default.yaml`、`tts_speakers.json` などをコピー；`app.yaml` で上書き可）。推論設定は `data/inference/config/`。ホストで Backend を直接起動する場合はプロジェクトルートの `config/`。

YAML 変更後は **該当サービスを再起動**；`.env` 変更後は **`docker compose up -d` でコンテナ再作成**。entrypoint が書いた推論設定が効かない場合は一時的に `VITOOM_OVERWRITE_CONFIG=1` で推論コンテナ再起動（§8 参照）。

### 12.1 テキスト LLM の VRAM 調整（`gpu_memory_utilization`）

vLLM テキストサービスが **予約する GPU メモリ割合**。範囲 `(0, 1]`。大きいほど VRAM 使用増。`config.runtime.backend: vllm` のテキストサービスのみ有効。

**Docker（永続ファイルを編集）**

テキスト設定（Text 初回起動後に生成）：

```text
data/inference/config/text.yaml
```

`config.runtime.vllm` 下で調整、例：

```yaml
config:
  runtime:
    vllm:
      gpu_memory_utilization: 0.75
```

- VRAM が厳しい：**下げる**（例 `0.5`～`0.7`）。
- より大きなコンテキストや重み：**上げる**必要があり、`max_model_len` も確認。
- 初回 `text.yaml` 生成時、entrypoint が約 14GiB / GPU 総 VRAM から比率を自動計算；**以降は自動更新されない** — GPU やモデル変更時は手動で調整。

編集後 Text を再起動：

```bash
docker compose -f docker-compose.inference.release.yml --profile text restart
```

**Web 管理**：管理者ログイン → 推論サービス管理 → テキスト（例 `text`）→ サービス設定で `config.runtime.vllm.gpu_memory_utilization` を編集（保存後 UI の指示で再起動）。

**ホスト開発**：`inference/config/ex_text.yaml` またはローカル `inference/config/text.yaml` の同パスを編集し、テキスト推論プロセスを再起動。

### 12.2 ストレージを S3 に変更

Backend のタスク／アップロード先は **`storage.default`**（`server` | `s3` | `oss`）で決まります。

**1) Backend**

`data/config/app.yaml` に追加（なければ作成）、例：

```yaml
storage:
  default: s3
  s3:
    endpoint: "https://s3.amazonaws.com"   # MinIO 等 S3 互換も可
    region: "ap-southeast-1"
    bucket: "your-bucket"
    access_key_id: "YOUR_ACCESS_KEY"
    secret_access_key: "YOUR_SECRET_KEY"
    public_base_url: "https://your-bucket.s3.ap-southeast-1.amazonaws.com"
```

`public_base_url` は公開 URL 用。バケットの公開方法と一致させること。

Backend 再起動：

```bash
docker compose up -d backend
```

**2) 推論**

`data/inference/config/inference.yaml` の `storage` も編集（`default: s3` と `storage.s3` を Backend と整合またはバケットポリシーに合わせる）し、該当推論コンテナを再起動。

### 12.3 デフォルトテキスト LLM の変更

**デフォルトチャットモデル**

`data/config/app.yaml`（Text 推論が動作し、`resources/models` に重みがあること）：

```yaml
agents:
  default_model: "your-model-name"
```

Backend 再起動。新規セッション／タスクに反映；既存セッションが古い `load_name` に紐づく場合は UI でモデル切替または新規セッション。

**テキスト推論を単一モデルに固定**

`text.yaml` でコメント解除して設定：

```yaml
config:
  fixed_model: "your-model-name"
```

例は `inference/config/ex_gemma_text.yaml`、`inference/config/ex_text.yaml` を参照。

保存後 Text コンテナ再起動；§12.1 の `gpu_memory_utilization` も見直し。

### 12.4 デフォルト動画生成モデルの変更

`data/config/app.yaml`：

```yaml
agents:
  tools:
    video_generator:
      default_model_name: "TurboWan2.1-T2V-1.3B-480P"
```

名前はシステムに登録された動画 **モデル名** と一致させること。Visual/Video 推論がデプロイ済みで、重みが配置されていること（デフォルト `resources/models`）。

### 12.5 対応モデルファミリ

動画：Wan 系、TurboWan 系  
音声：Qwen-tts、Qwen-asr、VoxCPM  
画像：SDXL、Qwen-Image、Z-Image、Flux、Flux.2 など主要 OSS 画像モデル  
言語：Qwen 系

### 12.6 リアルタイム Web 検索の有効化

https://www.tavily.com/ で API キー取得（枠内無料あり）、`.env` に `TAVILY_API_KEY` を設定。

### 12.7 推論加速のためのモデルキャッシュ

`data/inference/config/{image,video,text,qwen_asr,qwen_tts}.yaml` で `pipeline_cache_ttl_seconds` を 0 より大きくすると、該当推論サービスがモデルをキャッシュし、次回以降が大幅に高速化。**注意：** キャッシュ中は TTL 満了まで VRAM を保持します。

`data/inference/config/inference.yaml` の同項目を変更すると全推論サービスに適用できます。

```yaml
pipeline_cache_ttl_seconds: 1800
```
