# L44 中間層 MITM＋現有 Z3：找到新解

2026-09-19。依使用者要求執行至找到一組與44.txt不等價的新解後停止。
沒有更改正式main.py、Z3 encoding、verifier或既有等價規則。

## 候選來源與流程

1. 從零生成 L44 factor-4 固定content FKM necklaces。
2. 固定target k=2、eta=-1、content=(8,10,8,12)，完整粗PACF hash join
   得到16,928個候選配對。
3. seed=123 確定性打亂候選順序，用壓縮工作canonical key去重。
4. 每個粗工作12秒wall budget、每個binary Z3 check最多1000ms。
5. 第一個粗工作經中間層MITM產生5個子問題，原binary Z3依次回傳：
   UNSAT、UNSAT、UNSAT、UNSAT、SAT。
6. 完整verifier、target/content/compression重算、既有完整等價去重，
   最後用既有atomic writer追加44.txt。

44.txt只用於最後去重，沒有從既有PQCP生成種子。

## 結果

    L=44
    nonzero shifts=2,42
    nonzero PACS=-4
    a=11001100000101110000001101001010100001100101
    b=11110111100101000100001101100001000100110010

S(0)=88，S(2)=S(42)=-4，其餘非零位移全部為0；pqcp_objective=0。
除了既有verifier，也用直接的±1雙重迴圈獨立重算全部44個S(u)，完全一致。

- 從開始粗枚舉到取得SAT：7.062秒，包含粗枚舉及工作啟動。
- 此粗工作的MITM＋Z3 completion wall time：2.798秒。
- 正式結果由548組增加至549組；全部原始bytes是新檔案的完整前綴。
- 與原548個equivalence keys均不同。額外測試兩序列同時乘(-1)^i，
  其canonical key也不在原集合；這項額外審核未修改正式去重定義。
- 只證明本方法可以在L44找到新解。單次7.062秒不是平均time-to-solution，
  不能據此宣稱固定倍率加速或推論L68同樣容易。

## 可回溯資料

目錄 `results/middle_mitm_L44_until_new/`：

- `44.before.txt`：追加前完整備份。
- `run.json`：生成參數與target profiles。
- `current_job.json`：找到解的粗模式。
- `attempts.jsonl`：wall time、每個middle candidate及Z3結果。
- `sat_000001.json`、`new_solution.json`：完整新解及profile，保存於正式追加前。
- `audit.json`：獨立重算、原檔hash、等價比較及原內容未變的審核。

實驗入口：`experiments/search_middle_mitm_L44.py`。本次已成功停止，
沒有在找到新解後繼續搜尋。新測試只使用此解核對驗證／保存，未將它作為
任何搜尋初始化資料。

相關測試：34 passed（0.75秒），包括exact MITM、研究枚舉及新解保存測試。
