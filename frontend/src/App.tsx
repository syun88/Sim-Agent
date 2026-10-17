import { useEffect, useRef, useState } from 'react'
import type { Execution, Frame, Plan } from './types'

const examples = [
  'Move the red cube into the blue zone.',
  'Pick up the red cube and place it in the blue zone.',
  'Move the blue cube to the green target.',
]

export default function App() {
  const [instruction, setInstruction] = useState(examples[0])
  const [status, setStatus] = useState({ provider: '...', neo4j: '...', agent: '...', engine: 'genesis', description: 'Genesis World', graspMode: 'kinematic-grasp-hold' })
  const [frame, setFrame] = useState<Frame | null>(null)
  const [plan, setPlan] = useState<Plan | null>(null)
  const [history, setHistory] = useState<Execution[]>([])
  const [events, setEvents] = useState<any[]>([])
  const [verification, setVerification] = useState<any | null>(null)
  const [memory, setMemory] = useState<any>(null)
  const socketRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    const socket = new WebSocket('ws://127.0.0.1:8000/ws')
    socketRef.current = socket
    socket.onmessage = (message) => {
      const data: Frame = JSON.parse(message.data)
      setFrame(data)
      if (data.events.length) setEvents(prev => [...data.events, ...prev].slice(0, 80))
    }
    return () => socket.close()
  }, [])

  const refresh = async () => {
    const health = await (await fetch('/api/health')).json()
    const executions = await (await fetch('/api/executions')).json()
    const memory = await (await fetch('/api/memory')).json()
    setStatus({ provider: health.provider, neo4j: health.memory, agent: health.agent, engine: health.simulation_engine, description: health.simulation_description, graspMode: health.grasp_mode })
    setHistory(executions.slice().reverse())
    setMemory(memory)
  }

  useEffect(() => { refresh(); const timer = setInterval(refresh, 1500); return () => clearInterval(timer) }, [])

  const submit = async (forceFailure = false) => {
    await fetch('/api/tasks', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ instruction, randomize: false, force_failure: forceFailure }) })
  }

  const reset = async (randomize: boolean) => {
    await fetch(`/api/scene/reset?randomize=${randomize}`, { method: 'POST' })
  }

  const stop = async () => {
    if (frame?.current?.task_id) {
      await fetch(`/api/tasks/${frame.current.task_id}/stop`, { method: 'POST' })
    }
  }

  useEffect(() => {
    const planEvent = events.find(e => e.kind === 'plan_generated')
    if (planEvent) setPlan(planEvent.plan)
    const memoryEvent = events.find(e => e.kind === 'memory_retrieved')
    if (memoryEvent) setMemory((prev: any) => ({ ...prev, retrieved: memoryEvent.memories, count: memoryEvent.count }))
    const verificationEvent = events.find(e => e.kind === 'verification_completed')
    if (verificationEvent) setVerification(verificationEvent.verification)
  }, [events])

  return (
    <div className="min-h-screen bg-[#050910] text-slate-100 font-sans">
      <header className="flex items-center justify-between px-6 py-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-semibold">Embodied AI Agent</h1>
          <p className="text-xs text-slate-400">OpenArm v2 · {status.description} · Think. Act. Learn.</p>
        </div>
        <div className="flex gap-2 text-xs">
          <Badge label="Engine" value={frame?.engine ?? status.engine} tone="blue" />
          <Badge label="Grasp" value={frame?.grasp_mode ?? status.graspMode} tone="amber" />
          <Badge label="Robot" value="OpenArm v2" tone="blue" />
          <Badge label="AI" value={status.provider} tone={status.provider === 'ai&' ? 'green' : 'amber'} />
          <Badge label="Neo4j" value={status.neo4j} tone={status.neo4j === 'Connected' ? 'green' : 'amber'} />
          <Badge label="Agent" value={frame?.agent_state ?? status.agent} tone="blue" />
        </div>
      </header>

      <main className="grid grid-cols-12 gap-4 p-4">
        <section className="col-span-3 space-y-4">
          <Panel title="Natural Language Task">
            <textarea value={instruction} onChange={e => setInstruction(e.target.value)} rows={3} className="w-full rounded-lg bg-white/5 p-3 text-sm outline-none" />
            <div className="mt-3 flex gap-2">
              <button onClick={() => submit(false)} className="flex-1 rounded-lg bg-blue-600 py-2 font-medium hover:bg-blue-500">Execute</button>
              <button onClick={stop} className="rounded-lg bg-rose-600 px-3 py-2 hover:bg-rose-500">Stop</button>
              <button onClick={() => reset(false)} className="rounded-lg bg-white/10 px-3 py-2 hover:bg-white/20">Reset</button>
              <button onClick={() => reset(true)} className="rounded-lg bg-white/10 px-3 py-2 hover:bg-white/20">Random</button>
            </div>
            <div className="mt-3 space-y-2">
              {examples.map(example => <button key={example} onClick={() => setInstruction(example)} className="w-full rounded-md bg-white/5 px-3 py-2 text-left text-xs hover:bg-white/10">{example}</button>)}
              <button onClick={() => submit(true)} className="w-full rounded-md bg-rose-500/15 px-3 py-2 text-left text-xs text-rose-300 hover:bg-rose-500/25">Run recoverable fault demo</button>
            </div>
          </Panel>
          <Panel title="Agent Plan">
            {!plan ? <Empty text="No plan yet" /> : (
              <div className="space-y-2 text-xs">
                <Line label="Provider" value={plan.provider ?? ''} />
                <Line label="Latency" value={`${(plan.planning_latency ?? 0).toFixed(4)} s`} />
                {plan.memory_decision && <div className="rounded bg-emerald-500/10 px-2 py-1 text-emerald-300">{plan.memory_decision}</div>}
                {plan.steps.map((step, index) => <div key={index} className="rounded bg-white/5 px-2 py-1">{index + 1}. {step.skill} {step.object ?? step.target ?? ''}</div>)}
              </div>
            )}
          </Panel>
        </section>

        <section className="col-span-6">
          <Panel title="OpenArm v2 Physical Simulation">
            {frame?.image ? <img src={frame.image} alt="OpenArm v2 simulation" className="w-full rounded-lg border border-white/10" /> : <div className="h-[480px] rounded-lg bg-white/5" />}
            <div className="mt-3 text-xs text-slate-400">Sim time {frame?.sim_time.toFixed(2)}s · Grasp hold: {frame?.grasp_mode ?? status.graspMode} (kinematic, not contact)</div>
          </Panel>
        </section>

        <section className="col-span-3 space-y-4">
          <Panel title="Verification">
            {!verification ? <Empty text="Awaiting verification" /> : (
              <div className="space-y-2 text-xs">
                <div className={verification.success ? 'text-emerald-400' : 'text-rose-400'}>{verification.success ? 'SUCCESS' : 'FAILED'}</div>
                <Line label="Horizontal error" value={`${(verification.horizontal_position_error * 1000).toFixed(1)} mm`} />
                <Line label="Tolerance" value={`${(verification.tolerance * 1000).toFixed(1)} mm`} />
                <Line label="Released" value={String(verification.released)} />
                <Line label="Stable" value={String(verification.stable)} />
              </div>
            )}
          </Panel>
          <Panel title="Graph Memory">
            <div className="space-y-2 text-xs">
              <Line label="Backend" value={memory?.available ? 'Neo4j' : 'Local fallback'} />
              <Line label="Retrieved" value={String(memory?.count ?? 0)} />
              <div className="max-h-[160px] overflow-auto space-y-2">
                {(memory?.retrieved ?? []).map((item: any, index: number) => (
                  <div key={index} className="rounded bg-white/5 px-2 py-1">
                    {item.instruction} · {item.success ? 'success' : item.failure_category}
                  </div>
                ))}
              </div>
            </div>
          </Panel>
          <Panel title="Agent Events">
            <div className="h-[340px] space-y-2 overflow-auto text-xs">
              {events.map((event, i) => (
                <div key={i} className="rounded bg-white/5 px-2 py-1">{event.kind}</div>
              ))}
            </div>
          </Panel>
        </section>

        <section className="col-span-12">
          <Panel title="Execution History">
            <div className="grid gap-2 md:grid-cols-3">
              {history.map(item => (
                <div key={item.task_id} className="rounded-lg border border-white/10 p-3 text-xs">
                  <div className={item.success ? 'text-emerald-400' : 'text-rose-400'}>{item.success ? 'SUCCESS' : 'FAILED'}</div>
                  <div className="mt-1">{item.instruction}</div>
                  <div className="mt-1 text-slate-400">{item.total_duration.toFixed(2)}s · retries {item.retry_count}</div>
                </div>
              ))}
              {history.length === 0 && <Empty text="No executions yet" />}
            </div>
          </Panel>
        </section>
      </main>
    </div>
  )
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return <div className="rounded-xl border border-white/10 bg-white/[0.02] p-4"><h2 className="mb-3 text-sm font-medium">{title}</h2>{children}</div>
}
function Badge({ label, value, tone }: { label: string; value: string; tone: string }) {
  const colors = { green: 'bg-emerald-500/15 text-emerald-300', blue: 'bg-blue-500/15 text-blue-300', amber: 'bg-amber-500/15 text-amber-300' } as any
  return <span className={`${colors[tone]} rounded-full px-2 py-1`}>{label}: {value}</span>
}
function Empty({ text }: { text: string }) { return <div className="text-xs text-slate-500">{text}</div> }
function Line({ label, value }: { label: string; value: string }) { return <div className="flex justify-between"><span className="text-slate-500">{label}</span><span>{value}</span></div> }
