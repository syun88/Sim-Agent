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

## 未解決: 戻り経路の探索と「モデル距離 vs 実行」(2026-11-09 ライブ実測)

`plan_empty_arm_home` を探索にした: 高さ3種 × 経路形状3種 × **対象から離れる方向**（最近傍から離れる／
物体群の重心から離れる／左右）× 距離2種、それぞれ IK→スクリーン→**実測クリアランスで順位付け**
（最初の区間 → 全経路の順）。証拠に `considered` / `rejected_routes` を残す。513 tests pass。

ライブ（「赤を拾って青いゾーンに置いて」）の結果:

- 配置は成功・検証済み（`placement: True` / `undisturbed: True`）
- 後片付けで **PARK_COLLISION**（1.1 mm 変位、`openarm_left_ee_outer_finger|red_cube`）
- 候補は全部評価されたが **全候補が同じ値**（最初の区間 0.05 = ceiling、全経路 0.0308 m）

つまり「候補の選び方」では差が出ておらず、**モデル上のクリアランス（30.8 mm）と実際に触れた事実が
食い違っている**。ここは経路探索の追加では解決しない。次にやること（推奨順）:

1. モデル距離の不一致を突き止める（どの geom 対で 30.8 mm と出ているか、実行時に pad が本当に
   どこを通ったか）。1時間程度の調査。
2. 実行時の追従誤差を測り、経路判定を「モデル距離」ではなく**実測**で行う。
3. 対象物体の近傍を keep-out として経路から除外する（測れないなら、近づかない）。

## やらないこと

- 安全しきい値（20 mm / 3 mm / パッド上の余白）を「測ったから」と言って緩めない。これは方針。
- スキルの検証ゲート（接触・追従・保持）を外さない。外すと失敗が成功として記録される。
