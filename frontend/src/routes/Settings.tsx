import { useEffect, useState } from 'react';
import { Save } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { Field, Input } from '@/components/ui/Input';
import { useAppStore } from '@/lib/store';
import { useHealth } from '@/lib/hooks/useHealth';

const STORAGE_KEY = 'animeloom.settings';

interface LocalSettings {
  warehousePath: string;
  redisUrl: string;
  geminiKey: string;
  anthropicKey: string;
  civitaiKey: string;
}

const DEFAULTS: LocalSettings = {
  warehousePath: '/workspace/warehouse',
  redisUrl: 'redis://localhost:6379/0',
  geminiKey: '',
  anthropicKey: '',
  civitaiKey: '',
};

function load(): LocalSettings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULTS;
    return { ...DEFAULTS, ...JSON.parse(raw) };
  } catch {
    return DEFAULTS;
  }
}

export function Settings() {
  const health = useHealth();
  const pushToast = useAppStore((s) => s.pushToast);
  const [s, setS] = useState<LocalSettings>(load);

  // If backend tells us the warehouse path, seed it on first load.
  useEffect(() => {
    if (!localStorage.getItem(STORAGE_KEY) && health.data?.warehouse) {
      setS((prev) => ({ ...prev, warehousePath: health.data!.warehouse }));
    }
  }, [health.data?.warehouse]);

  function update<K extends keyof LocalSettings>(key: K, value: LocalSettings[K]) {
    setS((prev) => ({ ...prev, [key]: value }));
  }

  function save() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(s));
    pushToast({ kind: 'success', title: 'Settings saved (local)' });
  }

  return (
    <div className="p-8 flex flex-col gap-6 max-w-[760px]">
      <header>
        <h1 className="text-[28px]">Settings</h1>
        <p className="text-fg-hint text-[13px] mt-1">
          Stored in your browser. API keys are not sent to the backend.
        </p>
      </header>

      <Card>
        <h3 className="font-display font-semibold text-fg-1 text-[16px] mb-4">Paths</h3>
        <div className="flex flex-col gap-4">
          <Field label="Warehouse path" hint="Where models, LoRAs and outputs are cached on disk.">
            <Input
              value={s.warehousePath}
              onChange={(e) => update('warehousePath', e.target.value)}
              placeholder="/workspace/warehouse"
            />
          </Field>
          <Field label="Redis URL" hint="Celery broker for async LoRA training jobs.">
            <Input
              value={s.redisUrl}
              onChange={(e) => update('redisUrl', e.target.value)}
              placeholder="redis://localhost:6379/0"
            />
          </Field>
        </div>
      </Card>

      <Card>
        <h3 className="font-display font-semibold text-fg-1 text-[16px] mb-4">API keys</h3>
        <div className="flex flex-col gap-4">
          <Field
            label="Gemini API key"
            hint="Free at aistudio.google.com/apikey · 1500 req/day for story decomposition."
          >
            <Input
              type="password"
              value={s.geminiKey}
              onChange={(e) => update('geminiKey', e.target.value)}
              placeholder="AIzaSy…"
            />
          </Field>
          <Field
            label="Anthropic API key"
            hint="Optional · enables Claude-refined per-shot prompts (~$0.003 / story)."
          >
            <Input
              type="password"
              value={s.anthropicKey}
              onChange={(e) => update('anthropicKey', e.target.value)}
              placeholder="sk-ant-…"
            />
          </Field>
          <Field
            label="Civitai API key"
            hint="Optional · only needed if the HF anime LoRA download fails."
          >
            <Input
              type="password"
              value={s.civitaiKey}
              onChange={(e) => update('civitaiKey', e.target.value)}
              placeholder="paste token…"
            />
          </Field>
        </div>
      </Card>

      <div className="flex justify-end">
        <Button iconLeft={<Save size={14} />} onClick={save} glow>
          Save
        </Button>
      </div>
    </div>
  );
}
