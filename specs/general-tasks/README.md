# 汎用化: いろいろなタスク・データ取り・実機（2026-11-09 時点の計画）

目的: 特定の2キューブ・特定の手順・Genesis 専用の寄り道を外し、**タスクを足すのがデータ編集で済む**
状態にする。データ取り（模倣学習のアノテーション）と実機（実アーム）に同じ決定層を使う。

## できている土台

- `simulation/base.py` の `SimulationBackend` がエンジン非依存の契約（`object_pose` / `object_contacts` /
  `arm_qpos` / `step` / `frame` ...）。skills は Genesis を直接触らない。
- 決定は測定に置き換え済み: ステージ地点 / 配置 yaw / 接近 yaw・tilt・overfly / 積載可否 /
  regrasp 緩め量 / 手渡しの会議点。証拠は計画と一緒に返る。
- `scripts/export_lerobot_annotations.py` がエピソードを JSONL で出す（`box_space: world_aabb`、
  `entity_grounding`、`failure_observation`）。

## S1 語彙をデータにする（実装済み）

- `assets/catalog/scene_catalog.json` が**唯一の語彙**: 物体・ゾーン・呼び名（ja/en）・掴めるか・
  どちらの手に湧かせるか。
- 読む側: `planning/references.py`、`planning/language_fallback.py`、`planning/agent_planner.py`、
  `robots/spawn.py`、`GET /api/catalog` と Web UI ラベル。
- 受け入れ: カタログに `green_block` を1エントリ足すと、パーサ・プロンプト語彙・初期配置・ラベルが
  コード変更なしで追従する（`backend/tests/test_catalog.py`）。
- 手書きの正規表現（`red` が "red wine" に当たらない等）はカタログより優先。境界規則はデータで
  表現できないため。
- プロンプトは**文面を変えない**（語彙の一般化とプロンプト変更を同時にやると、生きたモデルの答えが
  変わって検証が落ちる。実測: `test_plan_ledger`）。

## S2 タスクをデータにする（実装済み・2026-11-09）

`assets/tasks/*.json` が**複合手順**を持つ。1ファイル = 1手順:

```json
{"id": "handover_then_place",
 "description": "pick with one hand, pass it to the other, then place it in a zone",
 "phrases": ["渡してから", "hand it over then"],
 "slots": {"object": "object", "zone": "zone", "giver": "arm", "receiver": "arm"},
 "requires": ["grasp", "handover", "place"],
 "goal": {"type": "place_in_zone", "object": "{object}", "target": "{zone}"},
 "actions": [{"op": "pick", "object": "{object}", "arm": "{giver}"},
             {"op": "handover", "object": "{object}", "arm": "{giver}", "receiver": "{receiver}"},
             {"op": "place", "object": "{object}", "target": "{zone}"}]}
```

`planning/task_templates.py` が読み、**スロットを実シーンで接地**してから提案エンベロープに展開する。
展開結果は通常の計画と同じく grounding・コンパイラ・契約を通る（**テンプレートは身体の能力を広げない**）。

- 拒否する場合: 未宣言スロット（`UNKNOWN_SLOT`）／未接地の物体・ゾーン・腕（`SLOT_NOT_OBSERVED`）／
  身体が持たない capability（`CAPABILITY_UNAVAILABLE`）／壊れた定義（`INVALID_TEMPLATE`）。
- モデルへは**指示が `phrases` に触れたときだけ**手順を提示する。それ以外のプロンプトは1バイトも
  変えない（実測: 無条件に1行足したら、生きたモデルの答えが変わって `test_plan_ledger` が落ちた）。
- 収録: `handover_then_place` / `pick_then_place` / `two_on_one`。
- 受け入れ: 新しい複合タスクは JSON 追加だけで展開・コンパイルできる（`test_task_templates.py` 11件）。
  未接地・能力不足・未知テンプレート名は拒否される。

残り: 決定論パス（モデル無し）からもテンプレートを引けるようにするか、は未着手。

## S3 実機バックエンド（未着手）

- `SimulationBackend` を実装する `RealArmBackend`（ROS2 ブリッジ or ベンダ SDK）。skills が触る
  `env.*` は25種なので、実装すべき面は既に列挙されている。`object_pose` は認識（マーカー/点群）、
  `object_contacts` は把持センサ（電流・圧力）に写す。
- 安全側は**実機で測り直す**: 接近クリアランス 20 mm、積載可否（パッド 40.5 mm）は Genesis の身体で
  測った値。実機では同じ query を実機の身体で走らせる。
- 受け入れ: 同じ指示・同じ計画で、実機が「把持→配置」を検証付きで完了する。

## S4 データ取りの汎用化（未着手）

- エクスポートを skill 非依存に: `skill` 名だけでなく `capability`（reach/grasp/carry/place/pour）と
  測定値（クリアランス、残差、regrasp 緩め量、会議点）を1レコードに。身体の別を `body` で持つ。
- 受け入れ: 同じスキーマで Genesis と実機のエピソードを混ぜて学習側に渡せる。

## 直した不具合: 目標がすでに成立しているとき（2026-11-09）

指示「青いキューブを右手で緑のゾーンに置いて」→ **FAILED: CONTRACT_VIOLATION**。原因は
`green_zone [0.48,-0.20]` と `blue_cube [0.5,-0.22]` が 2.8 cm（ゾーン半径 5.5 cm）で**すでに達成**で、
プランナが目標種別を `already_at_target` に書き換え、`return_home` だけの計画にしていたこと。
その結果 `allowed_object_motion` が空で凍結され、次に同じ物体へ触る計画が「要求されていない物体を
操作」として拒否された。加えて、`return_home` 自体が失敗すると**何もする必要がないタスクが失敗**した。

修正:

1. 目標種別は `place_in_zone` のまま（目標は*条件*であって状態ではない）、`already_satisfied: true` と
   判定根拠（距離・高さ差・許容値・座標・保持者）を残す。
2. 計画は**空**（`steps: []`）。ホーム復帰は要求に含まれないので、失敗してよい筋の動作を足さない。
3. `validate_plan` は「空の計画」を `already_satisfied` の placement に限って許可。
4. `TaskContract.freeze` の `allowed_object_motion` に**目標そのものの物体**を含める（要求は
   その物体についての要求なので、今回の試行に手順が無くても要求されている）。

検証: `test_command_planner.py`（空計画・種別・根拠・契約の許可集合）と、ライブ実測
（SUCCEEDED、`steps: []`、`already_satisfied: true`、`placement blue_cube: true` / `undisturbed
red_cube: true`）。契約チェックは緩めていない。

## 特定済み: 「モデルは 30 mm、実際は接触」の正体（2026-11-09 ライブ実測）

同じモデルに**計画経路**と**実行した関節列**の両方を問うた結果（同じ物体ポーズ、同じ geom 対）:

| 問い | 最小ギャップ | geom 対 | サンプル |
|---|---|---|---|
| 計画した経路 | **+30.78 mm** | `ee_base_link_left_00` / `red_cube_geom` | 67 / 68 |
| **実行した関節列** | **−4.72 mm（めり込み）** | 同じ | 142 / 143 |

つまり経路の選び方の問題ではなく、**実行した姿勢が計画した姿勢と違う**。実行時の EE は物体から最小
107 mm しか離れておらず、最後のサンプルでめり込んでいる＝**腕が計画した home 姿勢に到達せず、
途中の姿勢で止まり、その止まった姿勢が物体の中**だった（変位 1.1 mm、接触は外側フィンガー）。

補足: `park_empty_arm` は各 waypoint を `move_to_site(..., planned_joints=edge)` で動かし、成功を
受けて次へ進む。成功判定は「site が目標に着いたか」であって「計画した関節列に居るか」ではない。

### 実行クリアランスの時系列（2026-11-09、`70ac762` の計測）

各サーボサンプルで同じモデルに問うた結果（EE 位置 / モデル最小ギャップ）:

| EE (x, y, z) | ギャップ |
|---|---|
| 0.239, 0.181, 1.160 | +20 mm（ceiling） |
| 0.405, 0.154, 1.197 | +20 mm |
| 0.405, 0.154, 1.162 | +20 mm |
| 0.402, 0.154, 1.128 | +20 mm |
| 0.399, 0.154, 1.102 | **+5.3 mm** |
| 0.398, 0.154, 1.096 | **−2.8 mm** |
| 0.399, 0.153, 1.094 | **−4.7 mm** |

つまり最後の**下降**で、モデル上も単調に詰まってめり込む。計画した home 姿勢は 30.8 mm 空いているので、
**実行した最終姿勢が計画から ~35 mm ずれている**（＝その姿勢での位置決め誤差が home の余裕 30.8 mm を
上回った）。キューブは home の近傍 30.8 mm に置かれていた。

→ 本質は「home が物体に近すぎるのに、home を目標にした」こと。`nominal_clear` のしきい値 30 mm は
観測値 30.8 mm を「クリア」と判定してしまう。

### 修正: home の余裕を「その身体が実際に着地する誤差」から決める（実装済み・`92df911`）

`skills/motion_accuracy.py` を追加。着地誤差（指令姿勢 vs 実姿勢）を記録し、

```
required_clearance = max(記録された最大誤差, 実測デフォルト 0.035 m) * 1.5 + 0.01  (上限 0.12 m)
```

を home の必要余裕とする。固定 30 mm の判定を置き換えた。park の着地ごとに誤差を `record()` するので、
使うほどその身体の値になる（記録が無いときは 2026-11-09 の実測 35 mm を使う＝楽観側に倒さない）。

ライブ実測（同指示、`92df911`）:

| 項目 | 値 |
|---|---|
| 必要余裕 | **62.5 mm**（誤差 35 mm × 1.5 + 10 mm） |
| home の実測余裕 | 30.8 mm → **不足** |
| 判定 | `home_mode: scene_clearance_home`（home を諦めて上空待避） |
| 待避先 | z = 1.2397（home の 62 mm 上） |
| 結果 | **SUCCEEDED**（`placement: True` / `undisturbed: True`） |

つまり「当たらない戻り方」を、経路候補の追加ではなく **home を目標にしない判断**として実現した
（home は物体から 31 mm しか離れておらず、その身体の着地誤差 35 mm を吸収できない）。

残り: 着地誤差を park 以外のムーブでも記録する（いまは park の着地のみ）。keep-out は不要になった可能性が
高いが、物体が home の真下にある場合は未検証。

### 直し方（元のメモ）

1. **到達の検証**: 各 waypoint の後で `arm_qpos` と計画関節の差を測り、閾値超なら
   `HOME_SHORT_STOP`（衝突ではなく未達）として報告。衝突と未達を混ぜない。
2. **home の選び方**: 物体が home 近傍にあるときは home を諦めて上空待避（既存の `nominal_clear` の
   考え方を、到達できない場合にも広げる）。
3. 余力があれば keep-out: 直前まで持っていた物体の近傍を経路から除外する。

証拠は `post_task_homing.arms[].evidence` の `tightest`（計画経路）と `executed_path_clearance`
（実行関節列）に残る。

## やらないこと

- 安全しきい値（20 mm / 3 mm / パッド上の余白）を「測ったから」と言って緩めない。これは方針。
- スキルの検証ゲート（接触・追従・保持）を外さない。外すと失敗が成功として記録される。
