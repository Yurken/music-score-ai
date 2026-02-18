# music-score-ai

开源风格的 Python 项目：将音频转为节拍/和弦/低音线并导出 MIDI + MusicXML，同时支持 MusicXML/MIDI/简谱文本互转。

## 功能

- `POST /transcribe/audio`
  - 输入：`mp3/wav`
  - 输出：`tempo`, `beats`, `downbeats`, `chords`(时间戳), `bass_notes`, 导出文件路径（`midi`, `musicxml`）
- `POST /convert/score`
  - 输入：`MusicXML` 或 `MIDI`
  - 参数：`output_format` in `{musicxml, midi, jianpu_text}`
  - 输出：转换结果与文件路径
- `GET /health`

所有处理产物统一写入：`./outputs/{uuid}/`

## 项目结构

```text
music-score-ai/
├── src/
│   ├── api.py
│   ├── audio_io.py
│   ├── beat.py
│   ├── chords.py
│   ├── bass.py
│   ├── export.py
│   └── convert.py
├── tests/
│   ├── test_beat.py
│   ├── test_chords.py
│   └── test_export_musicxml.py
├── outputs/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 算法说明

- 节拍/速度：`librosa.beat.beat_track`
- 和弦：
  - 使用 chroma (`librosa.feature.chroma_cqt`) + 模板匹配 fallback
  - 支持 `maj / min / 7` 标签
- 低音线：
  - HPSS (`librosa.effects.hpss`) 后在低频段 (`30-220Hz`) 做 peak picking
  - 按节拍网格量化，生成 MIDI note 序列
- 导出：
  - MIDI：`mido`
  - MusicXML：`music21`
  - 简谱文本：`jianpu_text` 纯文本

## 安装

### 1) 创建虚拟环境

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) 安装依赖

```bash
pip install -U pip
pip install -r requirements.txt
```

> `mp3` 解码在某些环境可能依赖系统 `ffmpeg`。若 `mp3` 读取失败，优先改用 `wav`，或安装 ffmpeg。

## 运行服务

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

打开文档：`http://127.0.0.1:8000/docs`

## 使用 Docker

### 1) 构建镜像

```bash
docker build -t music-score-ai:latest .
```

### 2) 运行容器

```bash
docker run --rm \
  -p 8000:8000 \
  -v "$(pwd)/outputs:/app/outputs" \
  music-score-ai:latest
```

Windows PowerShell:

```powershell
docker run --rm `
  -p 8000:8000 `
  -v ${PWD}/outputs:/app/outputs `
  music-score-ai:latest
```

### 3) 使用 Docker Compose

```bash
docker compose up --build
```

说明：
- API 地址：`http://127.0.0.1:8000`
- 输出目录映射：宿主机 `./outputs` <-> 容器 `/app/outputs`

## API 示例

### 健康检查

```bash
curl http://127.0.0.1:8000/health
```

### 音频转谱

```bash
curl -X POST "http://127.0.0.1:8000/transcribe/audio" \
  -F "file=@/path/to/demo.wav"
```

示例响应（节选）：

```json
{
  "job_id": "6c5b8...",
  "tempo": 122.1,
  "beats": [0.51, 1.00, 1.49],
  "downbeats": [0.51],
  "chords": [{"time": 0.51, "chord": "C"}],
  "bass_notes": [{"time": 0.51, "duration": 0.49, "midi": 36, "velocity": 84}],
  "exports": {
    "midi": "outputs/6c5b8.../bass.mid",
    "musicxml": "outputs/6c5b8.../score.musicxml"
  }
}
```

### 谱面格式转换

```bash
curl -X POST "http://127.0.0.1:8000/convert/score" \
  -F "file=@/path/to/input.musicxml" \
  -F "output_format=jianpu_text"
```

## 测试

```bash
pytest -q
```

当前包含至少 3 个单元测试：
- 节拍 downbeat 解析
- 和弦时间轴格式
- MusicXML 输出基本校验

## 错误处理与日志

- 文件类型校验：不支持格式返回 `400`
- 最大上传大小限制：默认 `25MB`，超出返回 `413`
- 推理/转换超时：默认 `90s`，超时返回 `504`
- 依赖缺失提示：返回可读错误信息（例如缺少 `librosa` / `music21` / `mido`）

## 许可证

MIT
