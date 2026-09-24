# 本地即時同步口譯（Confucius4-R2T2 + Confucius4-T3PO）

麥克風 → 串流語音辨識 → 串流同步翻譯 → 網頁雙語字幕，**全部跑在這台 Mac 上**，
不經任何雲端服務。

- **語音辨識**：[Confucius4-R2T2](https://github.com/netease-youdao/Confucius4-R2T2)
  （2B，append-only 串流 ASR），用 llama.cpp 跑官方 GGUF，Metal 全層卸載。
- **翻譯**：[Confucius4-T3PO](https://github.com/netease-youdao/Confucius4-T3PO)
  （14B，真串流 READ/WRITE 同傳），用 llama.cpp 跑官方 GGUF。
- **介面**：T3PO 官方的 FastAPI + WebSocket 網頁，介面文字即時轉為繁體（上游檔案未修改）。

## 為什麼需要這個專案

兩個上游 repo 都只支援 CUDA：R2T2 的串流推論綁 vLLM，它附的 llama.cpp 後端只提供
Linux x86_64 + CUDA 的預編譯檔；T3PO 則要求 vLLM 服務。這個專案補上 Mac 專用的接縫，
**其餘完全重用上游程式碼**（上游放在 `vendor/`，只加了一個 `.env`）：

| 檔案 | 作用 |
|---|---|
| [`scripts/build_r2t2_native.sh`](scripts/build_r2t2_native.sh) | 為 macOS ARM + Metal 重編 `r2t2_llama` 的 pybind11 擴充模組。`native_ext.cpp` 本身可移植，只需補上 Metal shader 內嵌與 Mach-O 的 rpath。 |
| [`asr_server/r2t2_gguf.py`](asr_server/r2t2_gguf.py) | 用 llama.cpp/Metal 載入 R2T2（**預設路徑**）。 |
| [`asr_server/r2t2_mps.py`](asr_server/r2t2_mps.py) | 備用路徑：把 Transformers 模型包成 vLLM 的 `generate()` 介面跑在 MPS 上。不想編 C++ 時用這條。 |
| [`asr_server/ws_server_mac.py`](asr_server/ws_server_mac.py) | 以 R2T2 的 `/asr_stream_api_v1` 協定對外服務，所以 T3PO 的 ASR 客戶端零修改就能接。 |
| [`llm_proxy/min_tokens_proxy.py`](llm_proxy/min_tokens_proxy.py) | 在 llama.cpp 上還原 vLLM 的 `min_tokens` 語義。缺了它，T3PO 每次強制輸出都拿到空字串並被判讀為 WAIT，字幕永遠不會出現。也負責翻譯側的繁簡轉換。 |
| [`zh_script.py`](zh_script.py) | 繁簡轉換層。中文輸入輸出預設繁體（台灣用詞），但**模型內部維持簡體**。 |
| [`web_localized.py`](web_localized.py) | 把上游網頁的介面文字即時轉成繁體再送出，上游檔案零修改。 |

兩條 ASR 路徑都重用 R2T2 原生的串流邏輯（chunk 排程、前綴回滾、append-only 提交）。
接縫是上游 `r2t2_llama` 自己文件化的擴充點：串流方法只透過 vLLM 的 `generate()`
介面碰引擎，所以換掉引擎就好。

## 安裝

需要 [Homebrew](https://brew.sh)、[uv](https://docs.astral.sh/uv/)、Xcode command line
tools，以及約 17 GB 磁碟空間。

```bash
brew install llama.cpp cmake
bash scripts/setup.sh         # 上游 checkout（釘住 commit）+ venv + 依賴 + 編譯原生擴充
bash scripts/fetch_models.sh  # 權重，約 13 GB
```

[`scripts/setup.sh`](scripts/setup.sh) 會把兩個上游 repo clone 到 `vendor/` 並 checkout
到釘住的 commit —— README 裡所有實測數字都是對著那兩個 commit 量的。它接著會呼叫
[`scripts/build_r2t2_native.sh`](scripts/build_r2t2_native.sh) 編譯 ASR 的原生擴充模組
（約 3 分鐘）。

設定放在 [`config/t3po.env`](config/t3po.env)，不在 `vendor/` 裡面，所以上游更新不會蓋掉它。

> 不想編 C++ 的話可以設 `ASR_BACKEND=mps`，改用 PyTorch 路徑 —— 需要完整的 HF 權重
> （把 `fetch_models.sh` 裡的 `--exclude "*.safetensors"` 拿掉），代價是多 2.4 GB
> 記憶體、慢約 30%，且 chunk 要放寬到 480 ms。

## 啟動

```bash
bash scripts/start_all.sh
```

然後開 <http://127.0.0.1:8000/>，選方向（中→英 / 英→中）後開始說話。
停止：`bash scripts/stop_all.sh`。日誌在 `logs/`。

四個服務：

| 服務 | 埠 | 說明 |
|---|---|---|
| `llama-server` | 8010 | T3PO 14B GGUF，Metal |
| `llm_proxy` | 8011 | `min_tokens` 補墊層 |
| ASR | 8093 | R2T2 串流辨識，llama.cpp/Metal |
| Web UI | 8000 | T3PO 官方網頁 |

## 繁體中文

中文的輸入與輸出預設是**繁體（台灣用詞）**，但兩個模型都是在簡體上訓練的 —— R2T2 轉錄
簡體、T3PO 也譯成簡體，而 T3PO 的交錯歷史每個 chunk 都會重送一次，把繁體餵回去會讓它
偏離訓練分佈。所以轉換只發生在邊界，模型永遠看到簡體：

```
R2T2 轉錄（簡體）──s2twp──► 網頁字幕／終端機（繁體）
                                    │
T3PO 的 prompt ◄──tw2sp── T3PO 的歷史（繁體）
T3PO 輸出（簡體）──s2twp──► 網頁字幕（繁體）
```

`s2twp` / `tw2sp` 是互為反函數的一對，所以「顯示 → 模型 → 顯示」無損，包含用詞替換：
實測 `The new software needs more memory, and the printer driver must be updated.`
→ `新軟體需要更多記憶體，而且印表機驅動程式必須更新。`（而非軟件／內存／打印機／驅動程序）。

**網頁介面本身**（按鈕、標籤）也是繁體。上游的 `static/` 是簡體，但不直接改它 —— 否則
下次更新上游就會衝突。`web_localized.py` 在送出前轉換那三個靜態檔，並把路由插到上游
catch-all 靜態掛載之前（Starlette 依註冊順序比對，而那個掛載吃掉所有路徑）。路由是插進
上游的 app 而不是把它當子 app 掛載：它的 lifespan 跑著 session 回收任務，而子 app 收不到
lifespan 事件。介面的中文不參與任何比較（`value="zh2en"` 之類都是 ASCII），所以整檔轉換
不會改變行為。`<html lang>` 也一併從 `zh-CN` 改成 `zh-TW`。

設 `ZH_VARIANT=simplified` 可關閉，兩個方向都會回到簡體。

> 一個已知的小瑕疵：ASR 的轉換是逐增量套用的，所以用詞映射的詞組若剛好跨兩個增量就會
> 漏掉，那一次會顯示「軟件」而非「軟體」。翻譯路徑是整段轉換，不受影響。

## 記憶體需求（重要）

這是整個方案最硬的約束。24 GB 的機器上：

| | 佔用 |
|---|---|
| T3PO Q5_K_M（官方未出 Q4，最小就是這個） | 11.3 GB |
| R2T2 Q4_K_M + mmproj Q8_0 | 1.5 GB |
| Python 執行環境、VAD 等 | ~1 GB |
| **合計** | **~13.6 GB** |

選 GGUF 而非 PyTorch 路徑的主因就是這裡：R2T2 的權重從 3.8 GB（fp16 safetensors）
降到 1.46 GB，實測 ASR 服務的 GPU wired 記憶體增量為 **+1.31 GB**。

> 度量上的坑：PyTorch 的 MPS 配置是**可分頁**的，在 `ioreg` 的 GPU 計數器和
> `vm_stat` 的 wired 都看不到（`torch.mps` 自己報 3.82 GB）。llama.cpp 的 Metal
> buffer 則是 wired。這正好解釋了記憶體不足時 PyTorch 路徑不是直接失敗、而是
> 退化成 swap 慢死。

**執行期間不要同時跑其他吃 GPU 的工作。** 實測若有一個 ComfyUI 佔著 12 GB 與 99% GPU，
整條管線會掉進 swap，ASR 從每 chunk 200 ms 惡化到完全吐不出字。開始前先確認：

```bash
ioreg -r -d 1 -c IOAccelerator | grep -o '"In use system memory"=[0-9]*' | head -1
```

## 效能與調校

在 M4 Pro（14 核 / 24 GB）上實測，見 [`scripts/bench_asr.py`](scripts/bench_asr.py)。
單獨跑 ASR 時每個 chunk 的解碼成本：

| chunk 大小 | llama.cpp/Metal | Transformers/MPS |
|---|---|---|
| 160 ms（上游預設） | 112 ms（RTF 0.70） | 214 ms（RTF 1.34，跟不上） |
| **320 ms（本專案預設）** | **~112 ms（RTF 0.35）** | 155 ms（RTF 0.49） |

但單獨測不夠 —— 14B 翻譯模型會在同一顆 GPU 上 probe。端到端落後（音檔結束到收句）：

| chunk 大小 | llama.cpp/Metal | Transformers/MPS |
|---|---|---|
| 160 ms | 2.8 秒 | 跟不上 |
| **320 ms** | **0.5 秒** | 1.6 秒 |
| 480 ms | 0.5 秒（無進一步改善） | 0.05 秒 |

所以 GGUF 路徑用 320 ms，PyTorch 路徑得放寬到 480 ms。34 秒音檔端到端耗時 36.0 秒，
全程跟得上，翻譯在句子中途就開始輸出。

**成本會隨句子長度上升**，兩條路徑都一樣：R2T2 的串流設計每個 chunk 都把累積音訊整段
重送，音訊編碼器因此重做整段工作（約 `70 ms + 12 ms × 累積秒數`，PyTorch 路徑是
`115 + 10×`）。因此：

- VAD 在約 800 ms 靜音處切句（上游預設約 200 ms 太碎，實測會在一句話中間切三刀，
  每個接縫掉一個字）。
- 另有 `ASR_MAX_UTTERANCE_SEC`（預設 12 秒）作為保險，避免連續獨白讓成本失控。

### append-only 的代價

已提交的文字永遠不會回改，所以在右側上下文不足時定案的 token 就錯定了 —— 實測有一次
把「请做些」聽成「请坐下」（`qǐng zuò xiē` / `qǐng zuò xià`），譯文也就忠實地跟著錯。
因此 `ASR_UNFIXED_TOKEN_NUM` 設成 2（上游是 1），多保留一個 token 等右側上下文。
實測代價：

| | 首次出字 | 每 chunk 成本 | 收句時間 | 轉錄結果 |
|---|---|---|---|---|
| `=1`（上游） | 1.60s / 1.93s | 99–140 ms | 5.59s / 7.12s | 正確 |
| **`=2`（預設）** | 1.93s / 2.28s | 105–148 ms | 5.59s / 7.11s | 正確 |

也就是**首次出字晚約一個 chunk（330 ms）、解碼成本多約 6%，而收句時間完全不變**。
兩段測試音檔在兩種設定下都正確，所以這裡買的是保險而不是已證實的改善；
要最低延遲就設回 1。

可用環境變數：

| 變數 | 預設 | 說明 |
|---|---|---|
| `ASR_BACKEND` | `gguf` | `gguf`（llama.cpp/Metal）或 `mps`（Transformers） |
| `ASR_CHUNK_MS` | `320` | 串流 chunk 大小（`mps` 建議設 `480`） |
| `ASR_LOOKAHEAD_MS` | `160` | 第一個 chunk 的前視長度 |
| `ASR_MAX_UTTERANCE_SEC` | `12` | 單句上限，逾時強制收句 |
| `ASR_UNFIXED_TOKEN_NUM` | `2` | 提交前保留幾個 token 可改（上游是 1）。設 1 字幕早 ~330 ms 出現，但更容易提早定案 |
| `VAD_MIN_SILENCE_FRAME` | `80` | 切句所需靜音長度（約 10 ms/frame） |
| `ASR_GGUF_MMPROJ` | `mmproj-...-Q8_0.gguf` | 改成 f16 版（0.64 GB）可試更高編碼精度 |
| `ASR_NATIVE_LOG` | `logs/asr-native.log` | llama.cpp 的逐 chunk 訊息；設 `stderr` 可併回主日誌 |
| `LATENCY_MODE` | `native` | T3PO 品質-延遲檔位：`low` / `native` / `high` |
| `ZH_VARIANT` | `traditional` | 中文顯示用字；設 `simplified` 關閉轉換 |

## 驗證工具

```bash
.venv/bin/python scripts/selftest.py                       # 28 項端到端自我檢查
.venv/bin/python scripts/mic_test.py                       # 對著麥克風即時口譯（不用瀏覽器）
.venv/bin/python scripts/mic_test.py --list-devices        # 看有哪些輸入裝置
.venv/bin/python scripts/bench_asr.py                      # ASR 逐 chunk 成本
.venv/bin/python scripts/bench_asr.py --backend mps        # 換後端對照
.venv/bin/python scripts/ws_client_test.py                 # 只測 ASR 服務的協定
.venv/bin/python scripts/e2e_test.py                       # 整條管線（不用麥克風）
.venv/bin/python scripts/e2e_test.py --direction en2zh --audio your.wav
```

[`selftest.py`](scripts/selftest.py) 跑真實服務的真實協定（沒有 mock），退出碼是失敗數，
約 90 秒。改動任何東西之後先跑它。

[`mic_test.py`](scripts/mic_test.py) 走的是和網頁完全相同的伺服器路徑，只是把字幕印在
終端機。它的音量計就是用來回答麥克風問題的第一個問句：音訊到底有沒有進來？一排點代表
送進 socket 的是靜音，那問題在收音端（裝置或 macOS 的麥克風權限），不在模型。

預設會自動挑這台機器自己的麥克風。avfoundation 的裝置編號會在裝置連接/斷開時重排
（拔掉耳機，內建麥克風的索引就變了），所以 `--device` 收名稱片段比收數字可靠：
`--device airpods`。

> macOS 的麥克風權限是給執行程式的那個 app（終端機 / 編輯器），第一次跑會跳出授權對話框。
> 權限被拒時 ffmpeg 回報的也是 `Input/output error`，和裝置不存在同一個訊息。

`e2e_test.py` 會像瀏覽器那樣以 1 倍速餵音檔進 `/ws/simul-demo`，印出每個
ASR 增量與翻譯片段的到達時間。

## 已知限制

- **單一連線**：只有一份模型，ASR 推論是序列化的。這是個人用的口譯工具，不是多人服務。
- **llama.cpp 的音訊編碼器**在極輕的起音上不如 PyTorch 版穩健（上游因此保留了
  `stream_llama_hybrid` 模式）。實測的中英樣本兩者轉錄完全一致，但如果遇到起音被吃掉，
  可改 `ASR_GGUF_MMPROJ` 到 f16 版，或退回 `ASR_BACKEND=mps`。
- **句尾偶爾會多一小段碎片**（例如單獨一個 `Now,` 或 `.`）。這是強制 flush 在緩衝區
  近空時的產物，vLLM 的 `min_tokens=1` 也會有同樣行為。
- **中英雙向之外品質未經驗證**：T3PO 主要針對中↔英訓練，官方說其他方向未經嚴格評估。
- `latency_mode` 的 `low` / `high` 走 `logit_bias`，會送到 llama.cpp。預設的 `native`
  不帶這個參數，路徑最穩。
- R2T2 權重採網易的模型授權（非 Apache），見 `vendor/Confucius4-R2T2/MODEL_LICENSE`。
- `vendor/Confucius4-R2T2/r2t2_llama/{native,bin}/` 下是本機編譯產物（擴充模組與
  llama.cpp dylib），依上游設計必須放在套件目錄內；重編就跑 `scripts/build_r2t2_native.sh`。
