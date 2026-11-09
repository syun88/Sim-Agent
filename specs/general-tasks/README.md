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

## S3 身体の契約と適合検査（実装済み・2026-11-09）／実機実装は未着手

`simulation/conformance.py` が**身体に要求する操作**を宣言し、その契約を検査する。

- 読み取り専用（計画が何度でも問える）: `frame` / `object_pose` / `object_bounds` /
  `object_contacts` / `object_linear_speed` / `ee_pose` / `arm_qpos` / `home_arm_qpos` / `arm_limits` /
  `arm_kinematics`
- 命令: `set_arm` / `step` / `set_gripper` / `gripper_position` / `gripper_target`
- 形状だけでなく性質も見る: `step` が**実際に時間を進めるか**（進めない身体が「動いた」と報告できない）、
  `arm_kinematics.evaluate(q)` が姿勢を返すか、`home_arm_qpos` が `arm_qpos` と同じ長さか。

ライブ実測（`5d64ec8`、Genesis セル）: `run_genesis_gui.py` が起動時に自分を検査し
`/tmp/sim-agent-conformance.json` に書き出す →

```
robot: openarm_v2 | operations: 16 | problems: 0
```

同じ検査で通常タスクも SUCCEEDED（`placement: True` / `undisturbed: True`）。

実機に必要なのはこの 16 操作の実装だけ（`scripts/check_body_conformance.py --describe` で一覧）。
`object_pose` は認識、`object_contacts` は把持センサ、`step` は制御周期に対応する。**安全しきい値は
Genesis の身体で測った値なので、実機では同じ query を実機で走らせて測り直す**（`motion_accuracy` の
着地誤差も同様に実機で記録される）。

未着手: 実機ブリッジの実装そのもの、`body` を `real:<robot_id>` として書き出す経路。

## S4 データ取りの汎用化（実装済み・2026-11-09）

`scripts/export_lerobot_annotations.py` を **skill 非依存**にした（`schema: sim-agent.annotation/2`）:

- **`capability`**（`grasp` / `reach` / `carry` / `place` / `hand_over` / `throw` / `pour` / `home` /
  `actuate_gripper` / `actuate_joint` / `navigate`）を各サブタスクと各スキルに付与。未知の op は
  `capability: null` + `capability_unmapped: true` として**落とさない**。
- **`measurements`** = スキルが記録した測定値（`grasp_yaw_deg` / `grasp_tilt_deg` / `approach_overfly_m` /
  `measured_clearance_m` / `staging` / `placer_yaw_deg` / `relief_fraction` …）と、その出所スキル。
  **無い測定値は 0 ではなく不在**のまま。
- **`body`** = `{id, source}`。`world_state.robot.type` から取る。記録が無ければ
  `{id: null, source: 'unrecorded'}` で**推測しない**（sim と real を混同しない）。
- **`episode`** = サブタスク外の記録（後片付けの `home_mode` / 必要余裕 / 観測誤差、retry 数、status）。

実測（既存 387 レコード）: `capability` は `place 279 / grasp 38 / home 35 / hand_over 35`、
**145 レコードに測定値が入る**（以前は 0）。body は `openarm_v2`（`world_state.robot.type`）。

受け入れ: `backend/tests/test_annotation_export_body.py` 6件（capability / 未知 op / 測定値と出所 /
欠測は不在 / body 不明は不明 / episode）。テスト全体 530 passed。

残り: 実機レコードは `body` を `real:<robot_id>` として書き出す実装（`SimulationBackend` 実装側）。


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

着地誤差は**全ムーブ**で記録する（`control/motion.py` の `move_to_site` 出口、`d9d0a16`）。
接触を含むムーブ（`move_blocked`）は home の余裕には使わない — 掴む動作は意図的に止まるので、混ぜると
推定が歪む。記録は**バーを上げる方向にのみ効く**（`8b38a7b`）: 通常ムーブは 12 mm 程度で着地するが、
問題になった near-object 姿勢は 35 mm ずれていたので、記録の最大値が実測ベースラインを下回ると
30.8 mm を再び「クリア」と判定してしまう（実際に一度そうなった）。

ライブ（同指示、`8b38a7b`）:

- 1回目: **SUCCEEDED**、`home_mode: scene_clearance_home`、必要余裕 62.5 mm、観測誤差 35 mm
- 2回目: FAILED（`IK_FAILURE`）— 1回目でキューブが既に移動済みのため別の失敗。home 復帰の問題ではない

残り: 物体が home の**真下**にある場合の待避先は未検証。

### 直し方（元のメモ）

1. **到達の検証**: 各 waypoint の後で `arm_qpos` と計画関節の差を測り、閾値超なら
   `HOME_SHORT_STOP`（衝突ではなく未達）として報告。衝突と未達を混ぜない。
2. **home の選び方**: 物体が home 近傍にあるときは home を諦めて上空待避（既存の `nominal_clear` の
   考え方を、到達できない場合にも広げる）。
3. 余力があれば keep-out: 直前まで持っていた物体の近傍を経路から除外する。

証拠は `post_task_homing.arms[].evidence` の `tightest`（計画経路）と `executed_path_clearance`
（実行関節列）に残る。

## 収集ラウンド（2026-11-09、`b885c08`）

`traces-20261109/`（6 指示 × 3、ランダム配置）: **SUCCEEDED 6 / FAILED 12**。主因は
`move_to_target`/`approach_object` の IK_FAILURE（延べ 17 回）で、帰属は `PLANNING 10 / TRACKING 9 /
APPROACH_CONTACT 4`。

**分かったこと**: ランダム配置の受け入れは**把持の到達性しか見ていない**。開始時 `admitted_grasp_arm` は
全物体に付いているのに、配置目標（ゾーン・積み上げ先）は到達不能なことがある。→ reset の受け入れに
「その指示の目標」の到達性も含めるべき（未実装）。

**連続探索はライブで効いている**: approach 計画 14 件中、yaw 11 件・tilt 3 件が候補リスト外の値
（例 yaw -14.0625°, tilt -0.9375°）。14 件すべてに `refinements` が記録される。

**メカニズム予測器: NOT_ADOPTED**（18 例、40 例未満／PLANNING 0/4・TRACKING 0/3／確信度逆転 0.53 vs 0.66）。
学習部品を入れる根拠は今のところ無い。詳細は `~/dojo/lime-verify-20261025/collection-20261109/REPORT.md`。

## 配置目標の到達性（2026-11-09 追加・`41aa43b`）

`move_to_object` に**配置領域内の代替点探索**を入れた: 要求点の IK 余裕が足りなければ、領域（ゾーンの
広がり／支持面 − 物体）の中で ±2 cm, ±1 cm を測り、最良の点を採る。領域の外は試さない（別のタスクに
なるため）。測れないとき（IK が答えない）は従来どおり実行に委ねる。何も届かないときは
`DESTINATION_UNREACHABLE` で**測定値付きで拒否**する。

**が、収集成功率は動かなかった**（追試 2/12）。実測した理由:

```
move_to_target IK_FAILURE: Cannot carry blue_cube to [0.48, 0.18, 1.14]
  EE [0.2996, 0.1801, 1.175], position residual 0.0262 m (tolerance 0.012)
```

IK 探索は残差 ≤12 mm で「届く」と言うが、**実行は 26 mm 足りずに終わる**。home 復帰で見つけた
「探索と実行の差」と同じ型で、候補点を増やしても埋まらない。

### 着地誤差を含めた配置可否（実装済み・`1ba1bf1`）

`skills/placement_accuracy.py` — **配置に必要な精度を、落ちる場所の幾何から測る**:

```
probe 残差 + その身体の運搬着地誤差  ≤  どれだけ中心からずれてよいか
```

| 落ちる場所 | 必要精度（幾何から） |
|---|---|
| ゾーン | `ゾーン半径 − 物体半径`（実測 55 − 20 = **35 mm**） |
| 積み上げ | `支持面の半幅`（実測 23 mm、物体の大きさでは広がらない） |
| 測れない | `None` = **ゲートしない**（許可ではない） |

ゾーンの半径は MJCF の site size から読む（`scene.zone_site_radius`、`object_bounds(zone)` も返す
ようにした）。運搬の着地誤差は `motion_accuracy` がフェーズ別に記録し、**中央値**を使う（1 回の
ブロックされた運搬 50 mm で以後の配置が全部ゲートされるのを防ぐ）。**3 サンプル未満は床値**。

ライブ（`1ba1bf1`）: 同指示 SUCCEEDED、`placement: True` / `undisturbed: True`。

**テスト衛生**: 着地誤差はプロセス全体の状態なので、`conftest.py` の autouse フィクスチャで
テストごとにリセットする（実プロセスでも同じ危険があるため、推定にサンプル数を要求している）。

## 次の一手（2026-11-09 実測に基づく）

配置の可否は「場所の幾何」で測れるようになった（`placement_accuracy`）が、**成否の判定はまだ
コントローラの 12 mm 許容**で行われている。実測:

- 運搬は目的地の手前 **27 mm** で系統的に止まる（可動域の縁）
- ゾーンの自由度は **35 mm**（55 − 20）なので、27 mm のずれは**タスクとしては成功**
- それでも `IK_FAILURE` になるのは、12 mm で落としているため

→ `_carry_to` の成否を `placement_accuracy.required_accuracy` で判定する（配置ごとの要求精度）。
併せて、`approach_object` の失敗（8/12）は reset の受け入れが把持可能性を保証していないことの裏返し。

## 律速は上流に移った（2026-11-09 実測 `2bfbd9f`）

運搬の成否をタスク要求精度で判定する変更（`_carry_to(allow_error_m=...)`）は入れたが、直近ラウンドで
発動は **0 回**。失敗は接近・配置目標の上流で起きている:

| skill | error | 回数 |
|---|---|---|
| move_to_target | IK_FAILURE | 7 |
| approach_object | APPROACH_COLLISION | 5 |
| approach_object | IK_FAILURE | 4 |

原因（実測）:

1. **指示の手が配置の手と違う**: `blue_cube [0.54, 0.264]`, `admitted_grasp_arm: left` に「右手」指示
   → 右手では届かず接近探索が全滅。**測って早く言う**（「右手では届かない、左手なら届く」）べき。
2. **接近で物体を 6.7–8.0 mm 押す**: 20 mm スクリーンは通る。home と同じ「探索と実行の差」。

成功率の推移（同一6指示）: 6/18 → 2/12 → 4/12 → 3/12 → 2/12。**下流（配置・会議点・home）の改善は
律速ではない**。次は上流（手の受け入れ・接近クリアランスの実測化）に投資する。

## やらないこと

- 安全しきい値（20 mm / 3 mm / パッド上の余白）を「測ったから」と言って緩めない。これは方針。
- スキルの検証ゲート（接触・追従・保持）を外さない。外すと失敗が成功として記録される。
