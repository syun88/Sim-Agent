export type PlanStep = { skill: string; arm?: string; object?: string | null; target?: string | null }
export type Plan = { goal: Record<string, string>; arm?: string; steps: PlanStep[]; provider?: string; model?: string | null; provider_error?: string | null; planning_latency?: number; memory_decision?: string | null }
export type Verification = { success: boolean; horizontal_position_error: number; tolerance: number; released: boolean; stable: boolean }
export type Scene = { objects: { id: string; position: number[] }[]; targets: { id: string; position: number[] }[] }
export type Frame = { type: string; engine?: string; grasp_mode?: string; state: Scene; sim_time: number; image: string | null; events: any[]; agent_state: string; current: any | null }
export type Execution = { task_id: string; instruction: string; success: boolean; total_duration: number; retry_count: number; attempts: any[] }
