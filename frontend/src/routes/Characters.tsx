import { useState } from 'react';
import { Plus, UserCircle2, UploadCloud, Trash2, CheckCircle2 } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { Modal } from '@/components/ui/Modal';
import { Badge } from '@/components/ui/Badge';
import { Input, Textarea, Field } from '@/components/ui/Input';
import {
  useCharacters,
  useCreateCharacter,
  useDeleteCharacter,
} from '@/lib/hooks/useCharacters';
import { useAppStore } from '@/lib/store';

export function Characters() {
  const { data, isLoading } = useCharacters();
  const create = useCreateCharacter();
  const del = useDeleteCharacter();
  const pushToast = useAppStore((s) => s.pushToast);

  const [showModal, setShowModal] = useState(false);
  const [name, setName] = useState('');
  const [desc, setDesc] = useState('');
  const [imagePathsRaw, setImagePathsRaw] = useState('');

  function reset() {
    setName('');
    setDesc('');
    setImagePathsRaw('');
  }

  async function onCreate() {
    if (!name.trim()) return;
    const image_paths = imagePathsRaw
      .split('\n')
      .map((s) => s.trim())
      .filter(Boolean);
    if (image_paths.length === 0) {
      pushToast({ kind: 'error', title: 'Need at least one image path' });
      return;
    }
    try {
      await create.mutateAsync({ name: name.trim(), description: desc.trim(), image_paths });
      pushToast({ kind: 'success', title: `Queued LoRA training for ${name.trim()}` });
      reset();
      setShowModal(false);
    } catch (e) {
      pushToast({ kind: 'error', title: 'Failed to create character', body: String(e) });
    }
  }

  return (
    <div className="p-8 flex flex-col gap-6 max-w-[1100px]">
      <header className="flex items-end justify-between">
        <div>
          <h1 className="text-[28px]">Characters</h1>
          <p className="text-fg-hint text-[13px] mt-1">
            Train a LoRA per character — identity holds across every shot.
          </p>
        </div>
        <Button iconLeft={<Plus size={15} />} onClick={() => setShowModal(true)} glow>
          Add Character
        </Button>
      </header>

      {isLoading ? (
        <div className="text-fg-hint text-[13px]">Loading…</div>
      ) : !data || data.total === 0 ? (
        <Card className="p-12 flex flex-col items-center text-center gap-3">
          <UserCircle2 size={36} className="text-fg-3" strokeWidth={1.25} />
          <h3 className="text-fg-1 text-[16px] font-semibold">No characters yet</h3>
          <p className="text-fg-2 text-[13px] max-w-sm">
            Add 4-8 reference images of a character and AnimeLoom trains a LoRA so they look the
            same across every generation.
          </p>
          <Button iconLeft={<Plus size={15} />} onClick={() => setShowModal(true)} className="mt-2">
            Add your first character
          </Button>
        </Card>
      ) : (
        <div className="grid grid-cols-3 gap-4">
          {data.characters.map((c) => (
            <Card key={c.id} hover>
              <div className="flex items-start justify-between mb-3">
                <div
                  className="w-12 h-12 rounded-lg flex items-center justify-center shrink-0"
                  style={{
                    background:
                      'linear-gradient(135deg, var(--color-pink) 0%, var(--color-blue) 100%)',
                    boxShadow: 'var(--shadow-glow-pink)',
                  }}
                >
                  <UserCircle2 size={22} className="text-fg-inverse" strokeWidth={1.5} />
                </div>
                {c.has_lora ? (
                  <Badge status="completed">
                    <CheckCircle2 size={10} className="mr-1" />
                    LoRA
                  </Badge>
                ) : (
                  <Badge status="training">training</Badge>
                )}
              </div>
              <div className="font-display font-semibold text-fg-1 text-[15px]">{c.name}</div>
              <div className="text-fg-hint text-[12px] mt-1">
                {c.shot_count} shot{c.shot_count === 1 ? '' : 's'} · created{' '}
                {new Date(c.created).toLocaleDateString()}
              </div>
              <div className="flex justify-end mt-3">
                <button
                  onClick={() => {
                    if (confirm(`Delete ${c.name}?`)) del.mutate(c.id);
                  }}
                  aria-label={`Delete ${c.name}`}
                  className="text-fg-3 hover:text-red p-1 rounded transition-colors"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            </Card>
          ))}
        </div>
      )}

      <Modal
        open={showModal}
        onClose={() => {
          if (!create.isPending) setShowModal(false);
        }}
        title="Add Character"
        footer={
          <>
            <Button variant="ghost" onClick={() => setShowModal(false)} disabled={create.isPending}>
              Cancel
            </Button>
            <Button onClick={onCreate} disabled={create.isPending || !name.trim()}>
              {create.isPending ? 'Creating…' : 'Create + Train LoRA'}
            </Button>
          </>
        }
      >
        <div className="flex flex-col gap-4">
          <Field label="Name" htmlFor="char-name">
            <Input
              id="char-name"
              placeholder="sakura_haruno"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </Field>
          <Field label="Description" htmlFor="char-desc" hint="One sentence — hair, eye color, distinctive features.">
            <Textarea
              id="char-desc"
              rows={3}
              placeholder="Pink hair, green eyes, ninja headband"
              value={desc}
              onChange={(e) => setDesc(e.target.value)}
            />
          </Field>
          <Field
            label="Reference image paths"
            htmlFor="char-imgs"
            hint="One path per line. Use 4-8 images for best results."
          >
            <Textarea
              id="char-imgs"
              rows={4}
              placeholder={'/workspace/refs/sakura_01.png\n/workspace/refs/sakura_02.png'}
              value={imagePathsRaw}
              onChange={(e) => setImagePathsRaw(e.target.value)}
            />
          </Field>
          <div className="flex items-center gap-2 text-fg-hint text-[12px]">
            <UploadCloud size={14} />
            File-upload UI coming soon — paste paths for now.
          </div>
        </div>
      </Modal>
    </div>
  );
}
