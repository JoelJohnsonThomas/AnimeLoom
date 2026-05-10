import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Wand2 } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { Field, Select } from '@/components/ui/Input';
import { useGenerateAnime, useGenerateSequence } from '@/lib/hooks/useGenerate';
import { useAppStore } from '@/lib/store';
import type { Quality, Resolution } from '@/lib/types';

const SAMPLE_STORY = `A girl with pink hair walks through a cherry blossom forest at sunset. Petals fall around her as she stops at a wooden bridge and looks at the river below. The wind gently moves her hair.`;

const SAMPLE_SCRIPT = `SCENE: cherry blossom forest, sunset, warm light
CHAR: sakura — pink hair, green eyes, ninja headband

POSE: walking forward, slow pace
Sakura walks down the petal-strewn path, hair moving in the breeze.

POSE: looking up
She pauses and looks at the falling petals overhead.

POSE: hands clasped, smiling
She turns toward the camera, eyes bright.`;

type Mode = 'prose' | 'script';

export function ScriptEditor() {
  const [mode, setMode] = useState<Mode>('prose');
  const [text, setText] = useState(SAMPLE_STORY);
  const [script, setScript] = useState(SAMPLE_SCRIPT);
  const [quality, setQuality] = useState<Quality>('standard');
  const [resolution, setResolution] = useState<Resolution>('720p');
  const [fps, setFps] = useState(24);

  const navigate = useNavigate();
  const generateAnime = useGenerateAnime();
  const generateSequence = useGenerateSequence();
  const pushToast = useAppStore((s) => s.pushToast);
  const setActiveJob = useAppStore((s) => s.setActiveJob);

  const lines = useMemo(() => (mode === 'prose' ? text : script).split('\n'), [mode, text, script]);
  const scenes = useMemo(() => {
    if (mode === 'prose') return [];
    return script
      .split('\n')
      .filter((l) => l.trim().toUpperCase().startsWith('SCENE:'))
      .map((l, i) => ({ index: i + 1, label: l.replace(/^SCENE:\s*/i, '').slice(0, 40) }));
  }, [mode, script]);
  const chars = useMemo(() => {
    if (mode === 'prose') return [];
    return script
      .split('\n')
      .filter((l) => l.trim().toUpperCase().startsWith('CHAR:'))
      .map((l) => l.replace(/^CHAR:\s*/i, '').split('—')[0].trim());
  }, [mode, script]);

  async function onGenerate() {
    try {
      let jobId: string;
      if (mode === 'prose') {
        const job = await generateAnime.mutateAsync({
          text: text.trim(),
          quality,
          target_resolution: resolution,
          target_fps: fps,
        });
        jobId = job.job_id;
      } else {
        const job = await generateSequence.mutateAsync({ script: script.trim() });
        jobId = job.job_id;
      }
      setActiveJob(jobId);
      pushToast({ kind: 'info', title: 'Generation queued', body: `Job ${jobId.slice(0, 8)}…` });
      navigate(`/generate/${jobId}`);
    } catch (e) {
      pushToast({ kind: 'error', title: 'Failed to start job', body: String(e) });
    }
  }

  const pending = generateAnime.isPending || generateSequence.isPending;

  return (
    <div className="p-8 flex flex-col gap-6 max-w-[1200px]">
      <header className="flex items-end justify-between gap-4">
        <div>
          <h1 className="text-[28px]">Script Editor</h1>
          <p className="text-fg-hint text-[13px] mt-1">
            Plain story or structured script — both ship the same pipeline.
          </p>
        </div>
        <Button
          iconLeft={<Wand2 size={15} />}
          onClick={onGenerate}
          disabled={pending}
          glow
          size="lg"
        >
          {pending ? 'Queuing…' : 'Generate'}
        </Button>
      </header>

      {/* Mode toggle */}
      <div className="inline-flex rounded-md border border-border-subtle p-1 bg-bg-surface self-start">
        {(['prose', 'script'] as Mode[]).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`px-4 py-1.5 text-[12px] font-semibold rounded transition ${
              mode === m ? 'bg-pink text-fg-inverse' : 'text-fg-2 hover:text-fg-1'
            }`}
          >
            {m === 'prose' ? 'Prose' : 'Structured script'}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-[1fr_280px] gap-4">
        {/* Editor */}
        <Card padded={false} className="overflow-hidden">
          <div className="flex">
            {/* Line numbers */}
            <div className="bg-bg-darker px-3 py-4 border-r border-border-faint font-mono text-[12px] text-fg-3 select-none text-right leading-relaxed">
              {lines.map((_, i) => (
                <div key={i}>{i + 1}</div>
              ))}
            </div>
            <textarea
              className="flex-1 bg-bg-editor text-fg-1 font-mono text-[13px] p-4 leading-relaxed resize-none outline-none min-h-[420px]"
              value={mode === 'prose' ? text : script}
              onChange={(e) => (mode === 'prose' ? setText(e.target.value) : setScript(e.target.value))}
              spellCheck={false}
            />
          </div>
        </Card>

        {/* Sidebar */}
        <div className="flex flex-col gap-4">
          {mode === 'script' && (
            <Card>
              <h4 className="text-fg-1 text-[13px] font-semibold mb-3">Parsed</h4>
              <div className="flex flex-col gap-2 text-[12px]">
                <div>
                  <div className="text-fg-hint uppercase text-[10px] tracking-wider mb-1">
                    Scenes ({scenes.length})
                  </div>
                  {scenes.length === 0 ? (
                    <div className="text-fg-3">— none yet</div>
                  ) : (
                    scenes.map((s) => (
                      <div key={s.index} className="text-fg-2 truncate">
                        {s.index}. {s.label}
                      </div>
                    ))
                  )}
                </div>
                <div className="mt-2">
                  <div className="text-fg-hint uppercase text-[10px] tracking-wider mb-1">
                    Characters ({chars.length})
                  </div>
                  {chars.length === 0 ? (
                    <div className="text-fg-3">— none yet</div>
                  ) : (
                    chars.map((c, i) => (
                      <div key={i} className="text-fg-2">
                        {c}
                      </div>
                    ))
                  )}
                </div>
              </div>
            </Card>
          )}

          <Card>
            <h4 className="text-fg-1 text-[13px] font-semibold mb-3">Generation</h4>
            <div className="flex flex-col gap-3">
              <Field label="Quality">
                <Select value={quality} onChange={(e) => setQuality(e.target.value as Quality)}>
                  <option value="draft">draft (fast)</option>
                  <option value="standard">standard</option>
                  <option value="high">high (slow)</option>
                </Select>
              </Field>
              <Field label="Resolution">
                <Select
                  value={resolution}
                  onChange={(e) => setResolution(e.target.value as Resolution)}
                >
                  <option value="480p">480p</option>
                  <option value="720p">720p</option>
                  <option value="1080p">1080p</option>
                </Select>
              </Field>
              <Field label="FPS">
                <Select value={fps} onChange={(e) => setFps(Number(e.target.value))}>
                  <option value={8}>8</option>
                  <option value={16}>16</option>
                  <option value={24}>24</option>
                  <option value={30}>30</option>
                </Select>
              </Field>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
