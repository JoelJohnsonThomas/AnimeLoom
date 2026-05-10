import { useState } from 'react';
import { Download, Play, Film } from 'lucide-react';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { useJob } from '@/lib/hooks/useJob';
import { useAppStore } from '@/lib/store';

interface ShotItem {
  name: string;
  score: number;
  status: 'completed' | 'failed';
  videoPath?: string;
}

const SAMPLE_SHOTS: ShotItem[] = [
  { name: 'shot_001', score: 0.91, status: 'completed' },
  { name: 'shot_002', score: 0.84, status: 'completed' },
  { name: 'shot_003', score: 0.88, status: 'completed' },
];

function scoreColor(score: number) {
  if (score >= 0.85) return 'var(--color-green)';
  if (score >= 0.7) return 'var(--color-gold)';
  return 'var(--color-red)';
}

export function Results() {
  const activeId = useAppStore((s) => s.activeJobId);
  const job = useJob(activeId ?? undefined);
  const [hovered, setHovered] = useState<string | null>(null);

  // Pull shots from result if available; otherwise show sample.
  const result = job.data?.result as
    | { shots?: Array<{ shot_index?: number; quality_score?: number; status?: string; video_path?: string }>; final_video?: string }
    | undefined;
  const shots: ShotItem[] = result?.shots?.length
    ? result.shots.map((s, i) => ({
        name: `shot_${String((s.shot_index ?? i) + 1).padStart(3, '0')}`,
        score: s.quality_score ?? 0,
        status: (s.status === 'failed' ? 'failed' : 'completed') as 'completed' | 'failed',
        videoPath: s.video_path ?? undefined,
      }))
    : SAMPLE_SHOTS;
  const finalVideo = result?.final_video;

  const total = shots.length;
  const passed = shots.filter((s) => s.score >= 0.85).length;
  const avg = total > 0 ? shots.reduce((a, s) => a + s.score, 0) / total : 0;

  return (
    <div className="p-8 flex flex-col gap-6 max-w-[1200px]">
      <header>
        <h1 className="text-[28px]">Results</h1>
        <p className="text-fg-hint text-[13px] mt-1">
          {activeId
            ? `Job ${activeId.slice(0, 8)}…`
            : 'Showing sample shots. Generate from the Script Editor to see real output.'}
        </p>
      </header>

      <div className="grid grid-cols-4 gap-3">
        <Card>
          <div className="text-[11px] font-semibold uppercase tracking-wider text-fg-hint">
            Shots
          </div>
          <div className="font-display font-bold text-[28px] text-fg-1 mt-1.5 leading-none">
            {total}
          </div>
        </Card>
        <Card>
          <div className="text-[11px] font-semibold uppercase tracking-wider text-fg-hint">
            Passed
          </div>
          <div className="font-display font-bold text-[28px] text-green mt-1.5 leading-none">
            {passed}
          </div>
        </Card>
        <Card>
          <div className="text-[11px] font-semibold uppercase tracking-wider text-fg-hint">
            Avg Quality
          </div>
          <div
            className="font-display font-bold text-[28px] mt-1.5 leading-none"
            style={{ color: scoreColor(avg) }}
          >
            {avg.toFixed(2)}
          </div>
        </Card>
        <Card>
          <div className="text-[11px] font-semibold uppercase tracking-wider text-fg-hint">
            Threshold
          </div>
          <div className="font-display font-bold text-[28px] text-fg-1 mt-1.5 leading-none">
            0.85
          </div>
        </Card>
      </div>

      {/* Final video */}
      <Card
        className="border-pink/20"
        style={{
          background:
            'linear-gradient(135deg, rgba(255,107,157,0.08) 0%, rgba(108,143,255,0.06) 100%)',
        }}
      >
        <div className="flex items-center justify-between mb-3">
          <h3 className="font-display font-semibold text-fg-1 text-[18px]">Final Video</h3>
          {finalVideo ? (
            <Button variant="secondary" iconLeft={<Download size={14} />}>
              Download
            </Button>
          ) : null}
        </div>
        <div className="aspect-video rounded-lg border border-border-subtle bg-bg-deep flex items-center justify-center">
          {finalVideo ? (
            <video controls className="w-full h-full rounded-lg" src={finalVideo} />
          ) : (
            <div className="flex flex-col items-center gap-2 text-fg-3">
              <Film size={32} strokeWidth={1.25} />
              <span className="text-[12px]">
                {job.data?.status === 'running' ? 'Generating…' : 'No video yet'}
              </span>
            </div>
          )}
        </div>
      </Card>

      {/* Shot grid */}
      <div>
        <h3 className="text-[14px] font-semibold text-fg-1 mb-3">Shots</h3>
        <div className="grid grid-cols-4 gap-3">
          {shots.map((s) => (
            <Card
              key={s.name}
              hover
              onMouseEnter={() => setHovered(s.name)}
              onMouseLeave={() => setHovered(null)}
              padded={false}
            >
              <div className="aspect-square bg-bg-deep border-b border-border-faint flex items-center justify-center relative">
                {s.videoPath ? (
                  <video src={s.videoPath} className="w-full h-full object-cover" muted />
                ) : (
                  <Film size={24} className="text-fg-3" strokeWidth={1.25} />
                )}
                {hovered === s.name && (
                  <div className="absolute inset-0 flex items-center justify-center bg-bg-base/40">
                    <Play size={22} className="text-fg-1" fill="currentColor" />
                  </div>
                )}
              </div>
              <div className="p-3 flex items-center justify-between">
                <span className="font-mono text-[12px] text-fg-1">{s.name}</span>
                <Badge status={s.status}>
                  <span style={{ color: scoreColor(s.score) }}>{s.score.toFixed(2)}</span>
                </Badge>
              </div>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}
