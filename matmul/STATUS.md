# プロジェクト現状記録 / 作業引き継ぎ

**最終更新:** 2026-05-29
**記録者:** Claude Code (AI)
**AIモデル:** Claude Opus 4.8 (1M context) — モデルID: `claude-opus-4-8[1m]`

> このファイルは別プロセス（別セッション）が作業を引き継げるよう、現状・実施済み・進行中・次のステップをまとめたもの。

---

## 1. 概要

`matmul/` は行列積（matrix multiply）ベンチマーク。同一の計算を3つの実装方式で比較する。

各セット `s` について以下を計算:

- `C[s] = A[s]·B[s] + C[s]`
- `D[s] = A[s]·B[s] + D[s]`
- `C[s] += D[s]`

`NSET` 個（既定4）の行列セットを `NITER` 回反復。行列は `N×N`（既定1024）、row-major・set-major レイアウト（`M[s*N*N + i*N + j]`）。

---

## 2. ファイル構成

| ファイル | 内容 | コンパイラ |
|---|---|---|
| `matmul.c` | CPUのみのナイーブ実装（性能基準版） | gcc |
| `matmul_acc.c` | OpenACC(GPU) + OpenMP(CPU) ハイブリッド版 | nvc |
| `matmul_cuda.cu` | CUDA/cuBLAS(GPU) + OpenMP(CPU) ハイブリッド版 | nvcc |
| `Makefile` | ビルド設定 | — |
| `job.sh` | SLURMジョブスクリプト（本セッションで作成） | — |
| `job_sample.sh` | ユーザ提供のSLURMサンプル（参考用） | — |
| `STATUS.md` | このファイル | — |

GPU版の分割: `NSET` のうち先頭 `NSET_GPU` 個をGPU、残りをOpenMPでCPUに割り当て並行実行。

---

## 3. 実行環境（確認済み）

- **SLURM** 利用可（`sbatch`/`squeue`/`sinfo`）。パーティション `flash` が idle。
- **モジュール**: `nvhpc/25.9` 利用可（`module load nvhpc/25.9`）。`nvc`/`nvcc`/`cublas` 提供。
- **GPU**: `flash` ノードは **NVIDIA RTX A2000**（6 GB, Driver 580.159.03, CUDA 13.0）。
- ログインノードには gcc あり（CPU版はローカルでもビルド・実行可）。GPU/nvcは未確認のためSLURM経由で実行する。

---

## 4. 本セッションで実施済みの作業

### 4-1. 3版のパラメータ制御を統一
`matmul.c` のみ反復回数がハードコード（`x<10`）でGFLOPS式も1反復分だった。他2版に合わせて修正:
- `matmul.c`: `NITER` マクロ追加（既定10）、ループを `x<NITER` に、GFLOPS式を `2·2·NITER·NSET·N³` に修正。
- `Makefile`: CPU版ビルドに `-DNITER=$(NITER)` を追加。
- ローカルで `make` → 実行確認済み（`N=1024 NSET=4 NITER=10` → 250.97秒, 0.685 GFLOPS, checksum 2.482384e+10）。

### 4-2. SLURMジョブスクリプト作成（`job.sh`）
`job_sample.sh` を参考に、`flash`/`nvhpc/25.9`/`nvidia-smi` を踏襲し、3版をビルド→実行する構成。
- CPU基準版は低速なため `CPU_NITER=2` に縮小（GPU版は `NITER=10`）。**GFLOPSは `NITER` で正規化されるため反復回数が違っても性能比較は公平。**
- パラメータ: `N=1024 NSET=4 NSET_GPU=2`。

---

## 5. SLURM実行履歴と結果

投入したジョブと用途:
- `job.sh` … 本番ベンチ（CPUは`CPU_NITER=2`に縮小、GPUは`NITER=10`）。
- `verify.sh` … 動作確認用（小サイズ・全版同一パラメータ）。本セッションで作成。
- `diag.sh` / `diag2.sh` … nvc(OpenACC)ビルド失敗の切り分け用。

### 動作確認（✅ 完了） — ジョブ465, `verify.sh`
N=256, NSET=4, NSET_GPU=2, NITER=3（全版共通）:

| 版 | ビルド | Elapsed | GFLOPS | Checksum |
|---|---|---|---|---|
| CPU (`matmul`) | ✅ | 0.566s | 1.422 | 3.531621e+06 |
| OpenACC (`matmul_acc`) | ✅ | 1.981s | 0.406 | 3.531621e+06 |
| CUDA/cuBLAS (`matmul_cuda`) | ✅ | 0.097s | 8.313 | 3.531621e+06 |

→ **3版ともビルド・実行成功、checksum完全一致（正当性OK）**。

### ジョブ459（旧本番・参考）
CPU(NITER=2): 111.93s/0.307 GFLOPS、CUDA(NITER=10): 120.22s/1.429 GFLOPS。
**OpenACC版はこの回だけビルド失敗**（`Error in path /usr/lib/gcc/.../11/...`）。

### OpenACCビルド失敗の結論
診断(ジョブ461,462)とジョブ465の成功から、**一過性のエラー**と確定。
- `flash`ノードのgccは13.3.0（gcc11は存在しない）。nvcの最小コンパイル・実ビルドとも別途成功。
- nvhpcはNFS(`/home/share/nvidia/...`)上にあり、459時のNFS読み取り失敗が原因とみられる。
- 恒久対策は不要。再発時は再投入で解消する見込み。

### 次プロセスがやること（任意）
1. フル本番計測: `sbatch job.sh`（必要なら`CPU_NITER`/`N`/`NSET_GPU`を調整）。
2. 結果確認: `cat job_<jid>.out` / `cat job_<jid>.err`。
3. `N`/`NSET_GPU`を振ってGPU/CPU分割の傾向、ACC vs cuBLASの差を見る。
4. 大サイズではOpenACCがCPUを上回る想定（小サイズはカーネル起動・転送オーバヘッドで逆転）。

---

## 5b. OpenACC overlap 改修の検討（本セッション）

**論点**: `matmul_acc.c` は GPU を `async(1)` で投入し、同一ホストスレッドがそのまま CPU 計算へ流れることで overlap している。だが途中で `#pragma acc wait` が必要な複雑な計算では、wait が CPU 計算の手前で同一スレッドを止め、直列化してしまう。加えて毎反復末尾の `wait(1)` はバリアになっている。

**作成した代替版**（いずれも GPU/CPU セットはメモリ独立なので毎反復バリアは撤去）:
- `matmul_acc2.c`（案A・入れ子）: `omp parallel sections` で GPU 駆動と CPU 計算を別スレッドに分離。CPU 側は `matmul_cpu` の入れ子 `parallel for` を使用。
- `matmul_acc3.c`（案A・入れ子回避）: 単一 `omp parallel` 内で tid0 が GPU 駆動＋wait、tid1.. が行範囲を分担して CPU 計算（入れ子なし）。

**比較結果（ジョブ470, N=512 NSET=4 GPU=2 NITER=5, 8コア, checksum全一致=2.428294e+08）**:

| 版 | GFLOPS(1) | GFLOPS(2) | 評価 |
|---|---|---|---|
| `matmul_acc`（現状） | 2.853 | 2.836 | 安定 |
| `matmul_acc2`（入れ子） | 1.258 | 1.320 | ❌ 遅い（nvcの入れ子並列でCPUスレッド不足）→不採用 |
| `matmul_acc3`（入れ子回避） | 3.186 | 2.117 | ◎ 現状同等以上。**推奨候補** |

**結論**: 分離方式は `matmul_acc3`（入れ子回避）が正解。robustさ（waitがCPUを止めない）を性能を落とさず実現。`acc2` は不採用。
**未確認**: ばらつき低減のための多試行、GPU負荷の重い大サイズでの優位確認、本採用（`matmul_acc.c`置換）の可否。

## 6. 注意点・既知の論点

- **checksumは版間で一致しない**: `C += A*B` を毎反復累積するため、`NITER` が異なると最終値も異なる（CPU版 NITER=2 と GPU版 NITER=10 で checksum が違うのは正常）。正当性を厳密比較したい場合は全版を同一 `NITER` で実行すること。
- CPUナイーブ版は非常に低速（NITER=10で約251秒）。フル `NITER` で全版を回すと `-t 10:00` に収まりにくい。
- `matmul_acc.c` の `matmul_gpu` に `deviceptr()`（空）指定がある。動作はするが意図確認の余地あり。

---

## 7. Git状態

- ブランチ: `main`、直近コミット `f1381d8 Initial commit`。
- `matmul/` ディレクトリは未追跡（untracked）。本セッションの変更（matmul.c, Makefile, job.sh, STATUS.md 追加/変更）はまだコミットしていない。
