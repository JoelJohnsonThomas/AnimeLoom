import { Link } from 'react-router-dom';
import { Wand2, UserPlus, Film, Users, CheckCircle2, Activity } from 'lucide-react';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { SakuraOverlay } from '@/components/brand/SakuraOverlay';
import { useCharacters } from '@/lib/hooks/useCharacters';
import { useHealth } from '@/lib/hooks/useHealth';

interface RecentJob {
  id: string;
  shots: number;
  chars: number;
  score: number;
  status: 'completed' | 'failed' | 'running';
  time: string;
}

const RECENT_JOBS: RecentJob[] = [
  { id: 'chainsaw_ep01', shots: 5, chars: 3, score: 0.88, status: 'completed', time: '2 hours ago' },
  { id: 'haikyuu_intro', shots: 3, chars: 2, score: 0.92, status: 'completed', time: 'Yesterday' },
  { id: 'demo_story_01', shots: 2, chars: 1, score: 0.79, status: 'failed', time: '2 days ago' },
];

function StatTile({ icon: Icon, label, value }: { icon: typeof Wand2; label: string; value: string | number }) {
  return (
    <Card>
      <div className="flex items-start justify-between">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-wider text-fg-hint">
            {label}
          </div>
          <div className="font-display font-bold text-[28px] text-fg-1 mt-1.5 leading-none">
            {value}
          </div>
        </div>
        <Icon size={18} className="text-fg-hint mt-0.5" strokeWidth={1.5} />
      </div>
    </Card>
  );
}

export function Dashboard() {
  const characters = useCharacters();
  const health = useHealth();

  const charCount = characters.data?.total ?? 0;
  const shotsTotal = characters.data?.characters.reduce((acc, c) => acc + c.shot_count, 0) ?? 0;
  const trained = characters.data?.characters.filter((c) => c.has_lora).length ?? 0;

  return (
    <div className="p-8 flex flex-col gap-6 max-w-[1100px]">
      <header>
        <h1 className="text-[28px]">Dashboard</h1>
        <p className="text-fg-hint text-[13px] mt-1">
          {health.data
            ? `${health.data.name} v${health.data.version} · warehouse ${health.data.warehouse}`
            : 'Connecting to backend…'}
        </p>
      </header>

      {/* Hero — branded, with sakura overlay */}
      <Card
        className="relative overflow-hidden border-pink/20"
        style={{
          background:
            'linear-gradient(135deg, rgba(255,107,157,0.10) 0%, rgba(108,143,255,0.08) 100%)',
        }}
      >
        <SakuraOverlay count={6} />
        <div className="relative flex items-center justify-between gap-6 p-2">
          <div className="flex-1 min-w-0">
            <h2 className="font-display text-[22px] font-bold text-fg-1 leading-tight">
              Generate a new anime sequence
            </h2>
            <p className="text-fg-2 text-[13px] mt-2 max-w-[480px] leading-relaxed">
              Type a story, get consistent anime video. Character identity holds across every shot
              via trained LoRA adapters.
            </p>
            <div className="flex gap-3 mt-4">
              <Link to="/script">
                <Button variant="primary" iconLeft={<Wand2 size={15} />} glow>
                  Start from Script
                </Button>
              </Link>
              <Link to="/characters">
                <Button variant="secondary" iconLeft={<UserPlus size={15} />}>
                  Add Character
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </Card>

      {/* Stats */}
      <div className="grid grid-cols-4 gap-3">
        <StatTile icon={Users} label="Characters" value={charCount} />
        <StatTile icon={CheckCircle2} label="LoRA Trained" value={trained} />
        <StatTile icon={Film} label="Shots Generated" value={shotsTotal} />
        <StatTile icon={Activity} label="Avg Quality" value="0.87" />
      </div>

      {/* Recent jobs */}
      <section>
        <h3 className="text-[14px] font-semibold text-fg-1 mb-3">Recent jobs</h3>
        <Card padded={false}>
          <div className="grid grid-cols-[1fr_60px_1fr_60px_90px_100px] gap-3 px-4 py-2.5 border-b border-border-subtle bg-bg-deep text-[10px] font-semibold uppercase tracking-wider text-fg-hint">
            <span>Story</span>
            <span className="text-center">Shots</span>
            <span>Characters</span>
            <span className="text-right">Score</span>
            <span className="text-right">Status</span>
            <span className="text-right">Time</span>
          </div>
          {RECENT_JOBS.map((j, i) => (
            <div
              key={j.id}
              className={`grid grid-cols-[1fr_60px_1fr_60px_90px_100px] gap-3 px-4 py-3 items-center text-[12px] ${
                i < RECENT_JOBS.length - 1 ? 'border-b border-border-faint' : ''
              }`}
            >
              <span className="font-mono text-fg-1 truncate">{j.id}</span>
              <span className="text-center text-fg-2">{j.shots}</span>
              <span className="text-fg-2">{j.chars} characters</span>
              <span
                className="text-right font-semibold font-mono"
                style={{
                  color: j.score >= 0.85 ? 'var(--color-green)' : 'var(--color-gold)',
                }}
              >
                {j.score.toFixed(2)}
              </span>
              <div className="flex justify-end">
                <Badge status={j.status === 'completed' ? 'completed' : j.status === 'failed' ? 'failed' : 'running'}>
                  {j.status}
                </Badge>
              </div>
              <span className="text-right text-[11px] text-fg-3">{j.time}</span>
            </div>
          ))}
        </Card>
      </section>
    </div>
  );
}
