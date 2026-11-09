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

## S2 タスクをデータにする（未着手）

- `assets/tasks/*.json`: 目標の並び（`holding` / `placed` / `stacked` / `in_zone` / `poured`）と
  必要な capability を宣言。文法は据え置き、**複合タスクのテンプレート**をデータで足せるようにする。
- 受け入れ: 新しい複合タスク（拾う→渡す→棚に置く）を JSON 追加だけで計画できる。

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

## 見つけた不具合（未修正・2026-11-09 ライブ実測）

指示「青いキューブを右手で緑のゾーンに置いて」→ **FAILED: CONTRACT_VIOLATION: plan manipulates an
unrequested object**。

原因は計画側で、`green_zone` が `[0.48, -0.20]`、`blue_cube` が `[0.5, -0.22]` にあり
（距離 2.8 cm、ゾーン半径 5.5 cm）、**すでに目標を満たしている**。プランナは `already_at_target` を返し
`return_home` だけの計画を作るが、契約チェックはそれを「要求された最終条件の欠落／対象外の操作」として
拒否する。つまり「もう終わっている」を成功として扱う経路が無い。

直し方（案）: 目標が既に成立している場合は、計画を no-op として成功にし、証拠に
`already_satisfied` と判定根拠（位置・ゾーン半径）を残す。**契約チェックは緩めない**（他の経路の
取りこぼしを防いでいるため）。再現条件は上記の座標と指示だけで決まる。

## やらないこと

- 安全しきい値（20 mm / 3 mm / パッド上の余白）を「測ったから」と言って緩めない。これは方針。
- スキルの検証ゲート（接触・追従・保持）を外さない。外すと失敗が成功として記録される。
