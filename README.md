# PyVideoTracker-Annotator

一個基於 PySide6 的影片標註工具，專為物件追蹤和標註而設計。

## 功能特色

- 📹 支援多種影片格式 (MP4, AVI, MOV 等)
- 🎯 互動式物件標註 (拖拉框選)
- 🎮 鍵盤快捷鍵操作
- 📂 COCO-VID 格式導出
- 💾 自動暫存進度
- 🔄 復原/重做功能
- 👻 前一幀參考框顯示

## 系統需求

- Python 3.8+
- Windows 10/11 (已測試)
- 至少 4GB RAM

## 安裝

### 方式一：使用預編譯執行檔

1. 從 [Releases](https://github.com/yourusername/PyVideoTracker-Annotator/releases) 下載最新版本
2. 解壓縮到任意目錄
3. 執行 `PyVideoTracker-Annotator.exe`

### 方式二：從原始碼安裝

1. 克隆此儲存庫：
```bash
git clone https://github.com/yourusername/PyVideoTracker-Annotator.git
cd PyVideoTracker-Annotator
```

2. 安裝依賴：
```bash
pip install -r requirements.txt
```

3. 執行程式：
```bash
python main.py
```

## 使用方法

### 基本操作

1. **開啟影片**：檔案 → 開啟影片
2. **類別設定**：程式會自動尋找 `labels.txt` 檔案，或建立預設類別
3. **標註物件**：
   - 拖拉滑鼠建立新的標註框
   - 點擊選擇現有標註框
   - 拖拉移動標註框
   - 拖拉角落控制點調整大小
4. **導出結果**：檔案 → 導出 COCO-VID

### 鍵盤快捷鍵

- `A` / `←`：上一幀
- `D` / `→`：下一幀
- `空白鍵`：播放/暫停
- `W`：上一個類別
- `S`：下一個類別
- `Delete`：刪除選中的標註框
- `Ctrl+S`：儲存進度
- `Ctrl+Z`：復原

### 類別設定

在影片所在目錄建立 `labels.txt` 檔案，每行一個類別名稱：
```
median_nerve
artery
vein
```

## 輸出格式

程式導出 COCO-VID 格式的 JSON 檔案，包含：
- 影片資訊
- 幀資訊
- 標註資訊（包含追蹤 ID）
- 類別資訊

## 開發

### 建置執行檔

使用提供的批次檔進行打包：
```bash
build.bat
```

或手動使用 PyInstaller：
```bash
pyinstaller --clean PyVideoTracker-Annotator.spec
```

### 項目結構

```
├── main.py                    # 主程式
├── requirements.txt           # Python 依賴
├── build.bat                 # 打包腳本
├── PyVideoTracker-Annotator.spec  # PyInstaller 配置
├── hooks/                    # PyInstaller hooks
│   ├── hook-cv2.py
│   └── hook-numpy.py
└── README.md                 # 說明文件
```

## 已知問題

- 某些 H.265 編碼的影片可能無法正常開啟
- 極大的影片檔案可能導致記憶體不足

## 貢獻

歡迎提交 Issue 和 Pull Request！

## 授權

此專案採用 MIT 授權條款 - 詳見 [LICENSE](LICENSE) 檔案。

## 更新日誌

### v1.0.0
- 初始發布
- 基本標註功能
- COCO-VID 導出
- 鍵盤快捷鍵支援
