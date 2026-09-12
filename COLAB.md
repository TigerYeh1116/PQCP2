# Google Colab 執行方式

先在 Colab 選擇「執行階段 → 變更執行階段類型 → GPU」，再把專案上傳或 clone
到 Colab。以下假設專案目錄名稱是 `PQCP2`：

```python
%cd /content/PQCP2
!pip install -q -r requirements-colab.txt
```

確認 CUDA 後執行：

```python
import torch
assert torch.cuda.is_available(), "請先把 Colab 執行階段切換成 GPU"
print(torch.cuda.get_device_name(0))
```

```python
!python main.py --L 44 --minutes 10
```

若要啟用 Z3 completion：

```python
!python main.py --L 44 --minutes 10 --z3 --repair
```

預設後端現在是 `cuda`，也可明確指定 `--backend cuda`。原本三個
`--mps-...` 參數已改名為 `--cuda-...`；舊參數名稱仍保留為命令列別名，方便
沿用既有指令。第一次執行時，程式會偵測並捨棄不相容的 macOS C 執行檔，
然後使用 Colab 的 `cc` 自動編譯 Linux 版本。

舊的 PyTorch JSON checkpoint 可直接用 `--resume` 載入；其中的 MPS 裝置與
Metal observation 設定會在記憶體中轉換為 CUDA 與原生 PyTorch 實作，不會改寫
checkpoint，也不會更動搜尋數學。
