# Z3 分段預算與可接續待解佇列

## 行為

正式 `main.py` 的 PyTorch hybrid、C 與 Adam/resume 共用 completion bridge。
選擇使用 Z3 時，現在預設啟用分段排程，不改變主搜尋的 loss、SA、
鄰近操作、壓縮數學條件、verifier 或 L.txt 等價去重。
獨立 legacy Python runner 不使用這個 bridge，不在本次變更範圍。

對 L=4n、奇數 n≥3 的 factor-4 工作，現在預設先執行
[保持雙向壓縮的 C 矩形搜尋](rectangle_search.md)。這是另外的有限 C 輪，
不算 Z3 check；miss 後再進入下列 exact 排程。

每個 factor-4 job 分成「找下一個 factor-2 中間模式」與各模式的
「完整 binary lift」子工作。每一輪至多執行一次 Z3 `check()`：

1. 新子工作先給 1 秒。
2. UNKNOWN 子工作排回佇列尾端，下次預算增加為 2、4、8、16、32、60 秒。
3. 不同 job 輪流處理；同一 job 內，新的中間模式先得到短時間嘗試，
   再重試舊的困難分支。不能保證每輪會有新模式可用。
4. 到 60 秒仍 UNKNOWN，暫存為 `WAITING`；本次不立即無限重試，
   下次相同 L、seed 啟動會接續。也可以提高上限。
5. factor-2 直接 lift 與原本 radius completion 同樣保存 UNKNOWN、增加預算。
   radius completion 保留既有 guidance/beam；其耗時不包含在 Z3 timeout 內。

這是**重新建立 solver 後以更長預算重試**，不是暫停／恢復 Z3 的內部
搜尋堆疊，也不保存 learned clauses。因此不同輪的預算不能視為累積
的等效求解時間；重建和重複求解有成本，目前未證明能縮短首解時間。

## 精確性

發現過的中間模式會從後續枚舉中阻擋，避免重複枚舉，但其尚未解決的
binary lift 保存在獨立子工作中。只有：

- 中間枚舉真正 UNSAT；且
- **每個**已發現的 binary lift 都 UNSAT

才可判定該固定 factor-4 模式/content 無解。時間上限、中間模式數量
上限及 UNKNOWN 都不是無解證明，更不是某個 L 全域無解。
所有 SAT 仍經現有獨立驗證、target、content、compression 確認後，
經原 writer 去重寫入 L.txt。

輸出的 `LIFT_UNKNOWN work=enum next=queue` 也可能表示：找到了中間模式、
主動讓出本輪，完整 binary lift 尚未完成；不是每一筆 UNKNOWN 都是 timeout。
實際分支 timeout 原因保存在 checkpoint 的 `reason`。

## 保存與重啟

每個工作在入列**之前**已 atomic write 為 JSON：

```
checkpoints/z3/L68_seed123/<job-key>.json
```

保存原始候選、固定壓縮問題、target/content、發現過的中間模式、每個
分支的 READY/WAITING/UNSAT 狀態、下次預算、嘗試次數與輪轉順序。
schema version 與 checksum 用於偵測意外損壞，不使用 pickle。
同一固定壓縮問題即使來自不同完整序列，也共用原工作，不重新排隊。
現在也預設使用 [壓縮工作等價 key](compressed_job_equivalence.md)，
將模式、target、content 一起正確變換後去重；舊等價檔案原樣保留。
SAT/UNSAT 終結記錄保留以避免重做；停用的支線不會被偷偷恢復。

Ctrl+C 會等當前輪結束；尚未處理的工作已在磁碟上。下次從同一專案根目錄、
相同 L 與 seed 執行即自動恢復 Z3 佇列，不必先重新產生相同 FKM 候選。
這不代表 SA/GPU trajectory 也自動恢復；主搜尋仍遵循原來的 resume 機制。
同一 root/L/seed 的 checkpoint 只支援一個搜尋程序使用，勿同時啟動兩份。

工作先保存候選與正式結果，再標記 SAT；意外中斷最多重做該輪，
正式 writer 仍會去重。損壞 JSON/checksum 會明確報錯，不靜默覆寫。

## 參數與回退

- `--z3-slice-seconds 1`：新工作首次 `check()` 預算。
- `--z3-timeout 60`：每個分支單次 `check()` 最大預算；不限制整體搜尋時間。
- `--z3-timeout 0`：不設最大值，但仍從短預算逐次加倍，不是第一輪無限等待。
- `--z3-multilevel-max-models 16`：保留原本每 job 中間模式上限。
  達上限只暫停枚舉；既有 child 照常處理，重啟時提高此值可繼續找新模式。
- `--no-z3-staged`：回退原一次性求解，不載入分段工作。
- `--no-z3-compressed-dedup`：只停用壓縮工作對稱性去重，保留精確相同工作去重。
- `--no-z3`：完全停用 Z3；原有資料不刪除。

timeout 只限制 Z3 `check()`，建構、剪枝、guidance/beam、驗證與寫入另計；
因此 1 秒輪並非嚴格的 1 秒 wall-clock deadline，Ctrl+C 也不是立即終止。
實際 Z3 使用次數按完成的 check 計，不因同一工作多輪而漏算。

## 測試

`tests/test_staged_completion.py` 覆蓋真實小型 SAT、每輪 check 數、
短預算加倍、排程公平性、UNKNOWN 保留、枚舉耗盡不誤報 UNSAT、
中間模式上限續跑、重啟不重做已枚舉模式、Ctrl+C 路徑、資料損壞、
atomic replace 失敗不破壞原檔，以及正式結果驗證與去重。
測試只使用暫存目錄，不會改動專案既有 L.txt。
