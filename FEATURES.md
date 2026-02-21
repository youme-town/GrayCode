# GrayCode プロジェクト 機能仕様書

グレイコード構造化光法によるプロジェクタ-カメラ対応マップ構築パイプラインの全機能を記述する。

## 目次

1. [全体データフロー](#1-全体データフロー)
2. [パイプライン制御 (pipeline.py)](#2-パイプライン制御-pipelinepy)
3. [グレイコードパターン生成 (gen_graycode.py)](#3-グレイコードパターン生成-gen_graycodepy)
4. [パターン投影・撮影 (cap_graycode.py)](#4-パターン投影撮影-cap_graycodepy)
5. [グレイコードデコード (decode.py)](#5-グレイコードデコード-decodepy)
6. [C2P 補間 (interpolate_c2p.py)](#6-c2p-補間-interpolate_c2ppy)
7. [P2C 補間 (interpolate_p2c.py)](#7-p2c-補間-interpolate_p2cpy)
8. [C2P → P2C 変換 (gen_p2c_from_c2p.py)](#8-c2p--p2c-変換-gen_p2c_from_c2ppy)
9. [画像ワーピング (warp_image.py)](#9-画像ワーピング-warp_imagepy)

---

## 1. 全体データフロー

```text
gen_graycode.py
  │  入力: proj_height, proj_width, height_step, width_step
  │  出力: data/graycode_pattern/pattern_00.png ~ pattern_NN.png
  ▼
cap_graycode.py
  │  入力: パターン画像群 + カメラ (Canon EOS)
  │  出力: data/captured/capture_00.png ~ capture_NN.png
  ▼
decode.py
  │  入力: キャプチャ画像群 + プロジェクタ解像度
  │  出力: result_c2p.npy / .csv  (Camera→Projector)
  │        result_p2c.npy / .csv  (Projector→Camera)
  ▼
  ├──▶ interpolate_c2p.py
  │      入力: result_c2p.npy + カメラ解像度
  │      出力: result_c2p_compensated_{method}.npy / .csv / _vis.png
  │
  ├──▶ interpolate_p2c.py
  │      入力: result_p2c.npy + プロジェクタ解像度
  │      出力: result_p2c_compensated_delaunay.npy / .csv / _vis.png
  │
  └──▶ gen_p2c_from_c2p.py
         入力: result_c2p.npy
         出力: result_p2c.npy / .csv

warp_image.py  (補間済みマップを用いた画像変換)
  入力: 対応マップ + 画像
  出力: ワープ済み画像
```

---

## 2. パイプライン制御 (pipeline.py)

4 ステップを順次実行する高レベルオーケストレーション。

### 設定 (`GraycodePipelineConfig`)

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `proj_height` | `int` | (必須) | プロジェクタ画像の高さ (px) |
| `proj_width` | `int` | (必須) | プロジェクタ画像の幅 (px) |
| `height_step` | `int` | 1 | グレイコードブロックの垂直ステップ |
| `width_step` | `int` | 1 | グレイコードブロックの水平ステップ |
| `window_pos_x` | `int` | 0 | パターン表示ウィンドウの X 位置 |
| `window_pos_y` | `int` | 0 | パターン表示ウィンドウの Y 位置 |
| `run_capture` | `bool` | True | キャプチャステップの実行有無 |
| `run_decode` | `bool` | True | デコードステップの実行有無 |
| `run_interpolate` | `bool` | True | 補間ステップの実行有無 |

### 実行フロー

```text
[1/4] gen_graycode.main()      → パターン画像生成
[2/4] cap_graycode.main()      → 投影＆撮影 (run_capture=True 時)
[3/4] decode.main()            → デコード (run_decode=True 時)
[4/4] interpolate_c2p.main()   → C2P 補間 (run_interpolate=True 時, method="delaunay")
```

---

## 3. グレイコードパターン生成 (gen_graycode.py)

### 3.1 グレイコード (Gray Code) の概要

**グレイコード**（反射二進符号）は、隣接する値間でちょうど 1 ビットだけ異なる二進符号体系である。
$n$ ビットの自然二進数 $b$ に対し、対応するグレイコード $g$ は以下で得られる：

$$g = b \oplus (b \gg 1)$$

ここで $\oplus$ はビットごとの排他的論理和、$\gg$ は右シフトである。

構造化光法では、プロジェクタの各画素位置をグレイコードのビットパターンとして符号化し、
1 ビットずつ異なるパターン画像を順次投影する。
カメラで撮影した各画素の明暗系列からビット列を復元し、プロジェクタ座標を特定する。

### 3.2 ブロック化とパターン拡大

プロジェクタ解像度 $H \times W$ に対し、ステップサイズ $h_s, w_s$ を指定すると、
グレイコードは縮小された解像度で生成され、ブロック単位で拡大される。

**縮小解像度の計算：**

$$H_{gc} = \left\lfloor \frac{H - 1}{h_s} \right\rfloor + 1, \quad W_{gc} = \left\lfloor \frac{W - 1}{w_s} \right\rfloor + 1$$

OpenCV の `structured_light.GrayCodePattern.create(W_{gc}, H_{gc})` により、
$W_{gc} \times H_{gc}$ 画素分のパターン画像群が生成される。
パターン画像数は水平・垂直の符号化に必要なビット数の合計（$\lceil\log_2 W_{gc}\rceil + \lceil\log_2 H_{gc}\rceil$）枚である。

**ブロック拡大：**

各パターン画像を元のプロジェクタ解像度 $H \times W$ に拡大する。
出力画像の各画素 $(x, y)$ に対し：

$$\text{img}[y, x] = \text{pat}\!\left[\left\lfloor \frac{y}{h_s} \right\rfloor,\; \left\lfloor \frac{x}{w_s} \right\rfloor\right]$$

これは最近傍補間によるアップサンプリングに相当する。
ステップサイズ $> 1$ の場合、$h_s \times w_s$ 画素のブロックが同一のグレイコード値を持つ。

### 3.3 リファレンス画像

パターン画像群に加え、以下の 2 枚を追加する：

- **白画像**: 全画素 255（全面照射）
- **黒画像**: 全画素 0（無照射）

デコード時に有効画素を判定するための基準画像として使用する。

### 3.4 出力

`data/graycode_pattern/pattern_00.png` ～ `pattern_NN.png`
（パターン画像 + 白画像 + 黒画像、合計 $\lceil\log_2 W_{gc}\rceil + \lceil\log_2 H_{gc}\rceil + 2$ 枚）

---

## 4. パターン投影・撮影 (cap_graycode.py)

### 4.1 撮影プロセス

1. `data/graycode_pattern/pattern_*.png` を読み込む
2. OpenCV のフルスクリーンウィンドウにパターンを表示
3. **500 ms** 待機（プロジェクタの応答・安定化のため）
4. Canon EOS カメラで撮影

### 4.2 カメラ設定

Canon EDSDK を通じて以下のパラメータを設定する：

| パラメータ | 値 | 説明 |
|---|---|---|
| 絞り (AV) | 5 | 被写界深度の確保 |
| シャッター速度 (TV) | 1/15 秒 | パターンの十分な露光 |
| ISO 感度 | 100 | ノイズ低減 |
| 画質 | LJF (Large JPEG Fine) | 高解像度出力 |

### 4.3 色空間変換

撮影画像は RGB で取得され、グレースケールに変換して保存する：

$$I_{\text{gray}} = \text{cv2.cvtColor}(I_{\text{RGB}},\; \text{COLOR\_RGB2GRAY})$$

OpenCV の標準変換式：

$$I_{\text{gray}} = 0.299 R + 0.587 G + 0.114 B$$

### 4.4 出力

`data/captured/capture_00.png` ～ `capture_NN.png`（グレースケール画像）

---

## 5. グレイコードデコード (decode.py)

### 5.1 有効画素マスクの生成

キャプチャ画像群の末尾 2 枚を白画像 $I_w$ ・黒画像 $I_b$ として取り出す。
各カメラ画素 $(x, y)$ について：

$$\text{diff}(x, y) = I_w(x, y) - I_b(x, y)$$

$$\text{valid}(x, y) = \begin{cases} 1 & \text{if } \text{diff}(x, y) > T_{\text{black}} \\ 0 & \text{otherwise} \end{cases}$$

ここで $T_{\text{black}} = 50$ である。
この閾値処理により、プロジェクタ光が十分に届かない暗い領域や遮蔽領域を除外する。

### 5.2 グレイコードデコード

OpenCV の `GrayCodePattern` を使用して、有効な各カメラ画素のプロジェクタ座標を復元する。

**閾値パラメータ：**

- `setBlackThreshold(50)`: 暗い画素の判定閾値
- `setWhiteThreshold(5)`: 明暗の判定マージン

各有効画素 $(x_c, y_c)$ に対し、`getProjPixel(imgs, x_c, y_c)` を呼び出す。
この関数は全パターン画像の当該画素の明暗値を参照し、グレイコードのビット列を復号して、
縮小グレイコード空間上のプロジェクタ座標 $(g_x, g_y)$ を返す。

### 5.3 座標スケーリング

縮小空間の座標 $(g_x, g_y)$ をプロジェクタの実ピクセル座標 $(p_x, p_y)$ に変換する：

$$p_x = w_s \cdot \left(g_x + 0.5\right)$$

$$p_y = h_s \cdot \left(g_y + 0.5\right)$$

$+0.5$ の加算は、デコードされた座標がブロックのインデックス（左上基準）であるのに対し、
ブロックの**中心座標**を指すためのオフセットである。
例えば、$w_s = 4$ でデコード値 $g_x = 2$ の場合、$p_x = 4 \times 2.5 = 10.0$ となり、
ブロック $[8, 12)$ の中心を表す。

### 5.4 出力データ構造

#### C2P (Camera-to-Projector) マップ

各有効カメラ画素について 1 対 1 の対応を記録する：

$$\text{C2P}: (x_c, y_c) \mapsto (p_x, p_y)$$

- `result_c2p.npy`: `dtype=object` の配列、各要素は `((cam_x, cam_y), (proj_x, proj_y))`
- `result_c2p.csv`: `cam_x, cam_y, proj_x, proj_y`

#### P2C (Projector-to-Camera) マップ

C2P の逆引き辞書。同一プロジェクタ座標に複数のカメラ画素が対応する **1 対多** の関係を保持する：

$$\text{P2C}: (p_x, p_y) \mapsto \{(x_{c,1}, y_{c,1}),\; (x_{c,2}, y_{c,2}),\; \ldots\}$$

- `result_p2c.npy`: `dict` を `dtype=object` で wrap した 0-d 配列
  - キー: `(proj_x, proj_y)` タプル
  - 値: `[(cam_x, cam_y), ...]` のリスト
- `result_p2c.csv`: 1 対応 1 行で展開（`proj_x, proj_y, cam_x, cam_y`）

---

## 6. C2P 補間 (interpolate_c2p.py)

### 6.1 問題設定

デコードにより、カメラ画像上の**一部**の画素 $(x_{c,i}, y_{c,i})$ に対して
プロジェクタ座標 $(p_{x,i}, p_{y,i})$ が既知である。
目標は、カメラ画像の**全画素** $(x_c, y_c)$ に対してプロジェクタ座標を求めることである：

$$f: \mathbb{R}^2 \to \mathbb{R}^2, \quad (x_c, y_c) \mapsto (p_x, p_y)$$

2 つの補間手法が実装されている。

### 6.2 手法 1: Inpaint 法 (`interpolate_c2p_array`)

#### アルゴリズム

1. カメラ画像と同サイズの 2 枚のマップ $M_{p_x}, M_{p_y}$ を NaN で初期化する
2. 既知の対応点を座標値で直接書き込む：$M_{p_x}[y_c, x_c] = p_x$, $M_{p_y}[y_c, x_c] = p_y$
3. NaN 領域をマスクとして、`cv2.inpaint` で各チャネルを独立に穴埋めする

#### Telea アルゴリズム

OpenCV の `INPAINT_TELEA` を使用する。
Telea のアルゴリズムは **Fast Marching Method (FMM)** に基づき、
既知領域の境界から内側に向かって等距離面を伝播させながら値を補間する。

各未知画素 $q$ の値は、既知の近傍画素 $p$ からの距離 $d(p, q)$ に基づく重み付き平均で決定される。
Inpaint 半径は $r = 3.0$ ピクセルに設定されている。

**特徴：**

- 既知値を厳密に保持する（補間であり近似ではない）
- 凸包の外側にも値を外挿できる
- 座標値の滑らかさは局所的な拡散に依存するため、大きな欠損領域では精度が低下しうる

### 6.3 手法 2: ドロネー三角形分割法 (`interpolate_c2p_delaunay`)

#### 6.3.1 ドロネー三角形分割 (Delaunay Triangulation)

既知の対応点集合 $\{\mathbf{c}_1, \mathbf{c}_2, \ldots, \mathbf{c}_N\}$（カメラ座標）に対して、
カメラ座標平面上でドロネー三角形分割を構築する。

**空円性 (Empty Circumcircle Property):**
どの三角形の外接円の内部にも、他の入力点が含まれない。
これにより極端に細長い三角形が避けられ、補間の数値的安定性が高まる。

#### 6.3.2 重心座標による線形補間 (Barycentric Interpolation)

クエリ点 $\mathbf{q} = (q_x, q_y)$ が三角形 $\triangle(\mathbf{c}_a, \mathbf{c}_b, \mathbf{c}_c)$ の
内部にある場合、**重心座標** $(\lambda_a, \lambda_b, \lambda_c)$ を計算する：

$$\mathbf{q} = \lambda_a \mathbf{c}_a + \lambda_b \mathbf{c}_b + \lambda_c \mathbf{c}_c, \quad \lambda_a + \lambda_b + \lambda_c = 1, \quad \lambda_i \geq 0$$

対応するプロジェクタ座標は同じ重心座標で補間される：

$$f(\mathbf{q}) = \lambda_a \mathbf{p}_a + \lambda_b \mathbf{p}_b + \lambda_c \mathbf{p}_c$$

これは三角形内で**アフィン変換**に相当し、各頂点では既知の対応値を正確に再現する
（**補間**であり**近似**ではない）。

実装では `scipy.interpolate.LinearNDInterpolator` を使用する。

#### 6.3.3 凸包外部の処理 (Nearest Neighbor Extrapolation)

ドロネー三角形分割の凸包の外側に位置するクエリ点に対しては線形補間が定義できない。
**最近傍補間** を適用する：

$$f(\mathbf{q}) = \mathbf{p}_k, \quad k = \arg\min_i \|\mathbf{q} - \mathbf{c}_i\|_2$$

最も近い既知点のプロジェクタ座標をそのまま割り当てる。
内部的には `scipy.spatial.KDTree` による高速最近傍探索が使用される。
凸包外部では不連続な値の変化が生じうる。

### 6.4 出力

- `result_c2p_compensated_{method}.npy`: レガシー互換 `dtype=object` の `(N, 2, 2)` 配列
  - `[i, 0, :] = (cam_x, cam_y)`, `[i, 1, :] = (proj_x, proj_y)`
- `result_c2p_compensated_{method}.csv`: `cam_x, cam_y, proj_x, proj_y`
- `result_c2p_compensated_{method}_vis.png`: 可視化画像

---

## 7. P2C 補間 (interpolate_p2c.py)

### 7.1 問題設定

デコードにより、一部のプロジェクタ座標 $\mathbf{p}_i = (p_{x,i}, p_{y,i})$ に対して
対応するカメラ座標 $\mathbf{c}_i = (c_{x,i}, c_{y,i})$ が既知である。
目標は、プロジェクタ画像の**全画素** $(p_x, p_y)$ に対してカメラ座標を求めることである：

$$f: \mathbb{R}^2 \to \mathbb{R}^2, \quad (p_x, p_y) \mapsto (c_x, c_y)$$

### 7.2 1 対多対応の扱い

同一プロジェクタ座標に複数のカメラ座標が対応する場合（ステップサイズ $> 1$ の場合や
プロジェクタの 1 ピクセルが複数のカメラピクセルから観測される場合）、
全ての対応をそのままドロネー三角形分割の入力とする。

`LinearNDInterpolator` は重複入力点を持つ場合でも動作し、
重複点の値は三角形分割の過程で暗黙的に平均化される。

### 7.3 ドロネー三角形分割による補間

C2P 補間（セクション 6.3）と同じアルゴリズムを、定義域と値域を入れ替えて適用する：

| | C2P 補間 | P2C 補間 |
|---|---|---|
| 三角形分割の定義域 | カメラ座標平面 | プロジェクタ座標平面 |
| 補間する値 | プロジェクタ座標 $(p_x, p_y)$ | カメラ座標 $(c_x, c_y)$ |
| 出力グリッド | カメラ画像の全画素 | プロジェクタ画像の全画素 |
| 1 対多の扱い | 発生しない | 全対応を展開して入力 |

### 7.4 出力

- `result_p2c_compensated_delaunay.npy`: `(H*W, 4)` float32 配列 `[proj_x, proj_y, cam_x, cam_y]`
- `result_p2c_compensated_delaunay.csv`: `proj_x, proj_y, cam_x, cam_y`
- `result_p2c_compensated_delaunay_vis.png`: 可視化画像

---

## 8. C2P → P2C 変換 (gen_p2c_from_c2p.py)

### 8.1 処理内容

`result_c2p.npy` を読み込み、C2P 対応の逆引き辞書を構築する。

```python
for (cam_x, cam_y), (proj_x, proj_y) in c2p_array:
    p2c_dict[(proj_x, proj_y)].append((int(cam_x), int(cam_y)))
```

C2P では $(x_c, y_c) \mapsto (p_x, p_y)$ が 1 対 1 であるが、
異なるカメラ画素が同一のプロジェクタ座標にデコードされる場合があるため、
P2C は **1 対多** の対応となる。`defaultdict(list)` で全ての対応を保持する。

### 8.2 出力

- `result_p2c.npy`: `dict` を `dtype=object` で保存
- `result_p2c.csv`: 1 対応 1 行で展開

---

## 9. 画像ワーピング (warp_image.py)

PyTorch ベースの GPU 高速画像ワーピングモジュール。
対応マップを用いて Forward Warp（スプラッティング）と Backward Warp（サンプリング）を提供する。

### 9.1 座標系の定義

2 つの座標系を区別する：

| 座標系 | 用途 | ピクセル中心 | ピクセル $(i, j)$ の範囲 |
|---|---|---|---|
| **XY** | カメラ（ソース） | $(i,\; j)$ | $[i - 0.5,\; i + 0.5) \times [j - 0.5,\; j + 0.5)$ |
| **UV** | プロジェクタ（デスティネーション） | $(i + 0.5,\; j + 0.5)$ | $[i,\; i + 1) \times [j,\; j + 1)$ |

**ピクセルインデックスへの変換：**

- XY → ピクセル: $\text{idx} = \lfloor x + 0.5 \rfloor$
- UV → ピクセル: $\text{idx} = \lfloor u \rfloor$

### 9.2 Forward Warp（スプラッティング）

ソース画像（XY 空間）の各画素を、対応マップに基づいてデスティネーション画像（UV 空間）に配置する。
複数のソース画素が同一デスティネーション画素に写像される場合の集約方法を選択できる。

#### 9.2.1 Nearest Neighbor スプラッティング

各ソース画素 $(x, y)$ の値を、対応する 1 つのデスティネーション画素 $(\lfloor u \rfloor, \lfloor v \rfloor)$ に書き込む。
単純だが、デスティネーション画像に**穴**（対応のない画素）が生じやすい。

#### 9.2.2 Bilinear スプラッティング

各ソース画素を、デスティネーション座標 $(u, v)$ の周囲 4 近傍に分配する。

4 近傍のピクセル座標と重みは以下の通り：

$$x_0 = \lfloor u \rfloor, \quad y_0 = \lfloor v \rfloor$$

$$\Delta x = u - x_0, \quad \Delta y = v - y_0$$

| 近傍 | 座標 | 重み |
|---|---|---|
| 左上 | $(x_0,\; y_0)$ | $(1 - \Delta x)(1 - \Delta y)$ |
| 右上 | $(x_0 + 1,\; y_0)$ | $\Delta x (1 - \Delta y)$ |
| 左下 | $(x_0,\; y_0 + 1)$ | $(1 - \Delta x) \Delta y$ |
| 右下 | $(x_0 + 1,\; y_0 + 1)$ | $\Delta x \cdot \Delta y$ |

ソース画素の値 $I_s$ に重み $w$ を乗じた値 $w \cdot I_s$ を各近傍に蓄積する。

#### 9.2.3 集約方法 (Aggregation)

複数のソース画素が同一デスティネーション画素に重なる場合の処理：

| 方法 | 数式 | 説明 |
|---|---|---|
| **MEAN** | $\displaystyle I_d = \frac{\sum_i w_i \cdot I_{s,i}}{\sum_i w_i}$ | 重み付き平均 |
| **MAX** | $I_d = \max_i(I_{s,i})$ | 最大値 |
| **MIN** | $I_d = \min_i(I_{s,i})$ | 最小値 |
| **LAST** | $I_d = I_{s,\text{last}}$ | 最後の値で上書き |

### 9.3 Backward Warp（サンプリング）

デスティネーション画像（XY 空間）の各画素について、対応するソース画像（UV 空間）の座標を求め、
その位置の値をサンプリングする。

#### 9.3.1 サンプリンググリッドの構築

対応マップから、デスティネーション画像の各画素 $(x_d, y_d)$ に対する
ソース座標 $(u, v)$ のグリッドを構築する。
同一デスティネーション画素に複数の対応が存在する場合は平均化する：

$$\text{grid}_{uv}[y_d, x_d] = \frac{\sum_i (u_i, v_i)}{n}$$

対応のない画素には無効値 $(-10^6)$ を設定する。

#### 9.3.2 正規化座標への変換

PyTorch の `F.grid_sample` は $[-1, 1]$ の正規化座標を要求するため、以下の変換を行う：

$$\hat{x} = \frac{2u}{W_{uv}} - 1, \quad \hat{y} = \frac{2v}{H_{uv}} - 1$$

`align_corners=False` の設定に対応する正規化式である。

#### 9.3.3 `F.grid_sample` による補間

`torch.nn.functional.grid_sample` を使用して、正規化座標からソース画像の値を取得する。

**補間モード：**

- `bilinear`: 双線形補間
- `nearest`: 最近傍補間

**パディングモード（範囲外画素の処理）：**

| モード | 挙動 |
|---|---|
| `zeros` | 0 で埋める |
| `border` | 最も近い境界画素の値を使用 |
| `reflection` | 境界で反射して折り返す |

### 9.4 畳み込みベース Inpainting

Forward Warp / Backward Warp で生じた穴（対応のない画素）を反復的に埋める。

#### 9.4.1 カーネル

8 近傍（中心を除く）の均等重みカーネルを使用する：

$$K = \frac{1}{8}\begin{pmatrix} 1 & 1 & 1 \\ 1 & 0 & 1 \\ 1 & 1 & 1 \end{pmatrix}$$

重みカーネル（正規化なし）：

$$K_w = \begin{pmatrix} 1 & 1 & 1 \\ 1 & 0 & 1 \\ 1 & 1 & 1 \end{pmatrix}$$

#### 9.4.2 反復処理

各反復で以下を行う：

1. **穴画素の特定**: $\text{is\_hole} = (\text{mask} > 0.5)$

2. **近傍の有効値の合計と重みを計算**:

$$\text{sum}(x, y) = (I \cdot V) * K, \quad \text{weight}(x, y) = V * K_w$$

   ここで $I$ は現在の画像、$V$ は有効画素マスク、$*$ は畳み込みを表す。

3. **穴画素を近傍平均で埋める**:

$$I(x, y) \leftarrow \begin{cases} \displaystyle\frac{\text{sum}(x, y)}{\text{weight}(x, y)} & \text{if is\_hole}(x, y) \\ I(x, y) & \text{otherwise} \end{cases}$$

4. **マスクの縮小**（穴境界を 1 画素内側に進める）:

$$\text{mask} \leftarrow -\text{MaxPool2d}(-\text{mask},\; \text{kernel}=3)$$

   $-\text{MaxPool2d}(-\cdot)$ は MinPool に相当し、3×3 近傍に 1 つでも有効画素があれば穴を埋める。

この処理を指定回数繰り返すことで、穴の境界から内側へ向かって段階的に値が伝播する。

---

## 10. 可視化

C2P / P2C の補間結果を RGB 画像として可視化する。
各画素の色を対応する座標値から決定する：

$$\text{RGB}(x, y) = \left(\; f_x(x,y) \bmod 256,\;\; f_y(x,y) \bmod 256,\;\; 128 \;\right)$$

ここで $(f_x, f_y)$ は補間されたマッピング先の座標である。
R チャネルが水平座標、G チャネルが垂直座標を表し、B チャネルは固定値 128 とする。
座標値を 256 で割った剰余を取ることで、周期的な色パターンが生じ、
マッピングの空間的な連続性と異常を目視で確認できる。
