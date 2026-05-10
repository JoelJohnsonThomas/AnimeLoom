import { useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { Image as ImageIcon, Video, Sparkles, Layers, Wand2 } from 'lucide-react';
import { Card } from '@/components/ui/Card';
import { Badge, type BadgeStatus } from '@/components/ui/Badge';
import { Progress } from '@/components/ui/Progress';
import { Button } from '@/components/ui/Button';
import { useJob } from '@/lib/hooks/useJob';
import { useAppStore } from '@/lib/store';
import type { JobStatusName } from '@/lib/types';

interface Phase {
  icon: typeof Wand2;
  name: string;
  range: [number, number]; // progress range [start, end]
}

const PHASES: Phase[] = [
  { icon: ImageIcon, name: 'SDXL Keyframes', range: [0, 0.25] },
  { icon: Video, name: 'Wan2.2 I2V', range: [0.25, 0.6] },
  { icon: Sparkles, name: 'Face Lock', range: [0.6, 0.85] },
  { icon: Layers, name: 'Post & Assembly', range: [0.85, 1.0] },
];

function phaseStatus(progress: number, p: Phase): 'completed' | 'running' | 'pending' {
  if (progress >= p.range[1]) return 'completed';
  if (progress >= p.range[0]) return 'running';
  return 'pending';
}

function jobStatusToBadge(s: JobStatusName): BadgeStatus {
  switch (s) {
    case 'completed':
      return 'completed';
    case 'running':
      return 'running';
    case 'failed':
      return 'failed';
    default:
      return 'pending';
  }
}

export function Generate() {
  const { jobId: paramId } = useParams();
  const navigate = useNavigate();
  const activeId = useAppStore((s) => s.activeJobId);
  const pushToast = useAppStore((s) => s.pushToast);
  const jobId = paramId ?? activeId ?? undefined;
  const job = useJob(jobId);

  // Auto-navigate to results when complete
  useEffect(() => {
    if (job.data?.status === 'completed') {
      pushToast({
        kind: 'success',
        title: 'Generation complete',
        body: 'Your anime sequence is ready in Results.',
      });
      const t = setTimeout(() => navigate('/results'), 1200);
      return () => clearTimeout(t);
    }
    if (job.data?.status === 'failed') {
      pushToast({
        kind: 'error',
        title: 'Generation failed',
        body: job.data.error ?? 'Unknown error',
      });
    }
  }, [job.data?.status, job.data?.error, navigate, pushToast]);

  if (!jobId) {
    return (
      <div className="p-8 max-w-[1100px]">
        <header>
          <h1 className="text-[28px]">Generate</h1>
          <p className="text-fg-hint text-[13px] mt-1">
            No active job. Start one from the Script Editor.
          </p>
        </header>
        <div className="mt-6">
          <Button iconLeft={<Wand2 size={15} />} onClick={() => navigate('/script')}>
            Go to Script Editor
          </Button>
        </div>
      </div>
    );
  }

  const progress = job.data?.progress ?? 0;
  const status = job.data?.status ?? 'pending';

  return (
    <div className="p-8 flex flex-col gap-6 max-w-[1100px]">
      <header className="flex items-end justify-between gap-4">
        <div>
          <h1 className="text-[28px]">Generate</h1>
          <p className="text-fg-hint text-[13px] mt-1">
            Job <span className="font-mono text-fg-2">{jobId.slice(0, 8)}…</span>
          </p>
        </div>
        <Badge status={jobStatusToBadge(status)}>{status}</Badge>
      </header>

      <Card>
        <Progress value={progress} label="Overall progress" />
      </Card>

      <div className="grid grid-cols-4 gap-3">
        {PHASES.map((p) => {
          const s = phaseStatus(progress, p);
          const Icon = p.icon;
          return (
            <Card
              key={p.name}
              className={
                s === 'running'
                  ? 'border-pink/30 shadow-[var(--shadow-glow-pink)]'
                  : s === 'completed'
                    ? 'border-green/30'
                    : ''
              }
            >
              <div className="flex items-center gap-2 mb-2">
                <Icon
                  size={16}
                  strokeWidth={1.75}
                  className={
                    s === 'running' ? 'text-pink' : s === 'completed' ? 'text-green' : 'text-fg-3'
                  }
                />
                <Badge status={s === 'completed' ? 'completed' : s === 'running' ? 'running' : 'neutral'}>
                  {s}
                </Badge>
              </div>
              <div className="text-fg-1 text-[14px] font-semibold">{p.name}</div>
            </Card>
          );
        })}
      </div>

      {status === 'failed' && job.data?.error && (
        <Card className="border-red/30">
          <h4 className="text-red text-[13px] font-semibold mb-2">Error</h4>
          <pre className="text-fg-2 text-[12px] whitespace-pre-wrap break-words">
            {job.data.error}
          </pre>
        </Card>
      )}

      {status === 'completed' && (
        <Card className="border-green/30">
          <h4 className="text-green text-[13px] font-semibold mb-2">Ready</h4>
          <p className="text-fg-2 text-[13px] mb-3">
            Generation complete. Heading to Results…
          </p>
          <Button onClick={() => navigate('/results')}>Open Results</Button>
        </Card>
      )}
    </div>
  );
}
