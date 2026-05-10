import { useAppStore } from '@/lib/store';
import { X, CheckCircle2, AlertCircle, Info } from 'lucide-react';
import { cn } from '@/components/utils';

const KIND_ICON = {
  info: Info,
  success: CheckCircle2,
  error: AlertCircle,
};

const KIND_COLOR = {
  info: 'text-blue',
  success: 'text-green',
  error: 'text-red',
};

export function Toaster() {
  const toasts = useAppStore((s) => s.toasts);
  const dismiss = useAppStore((s) => s.dismissToast);

  return (
    <div className="fixed bottom-6 right-6 z-50 flex flex-col gap-2 max-w-sm">
      {toasts.map((t) => {
        const Icon = KIND_ICON[t.kind];
        return (
          <div
            key={t.id}
            className={cn(
              'flex items-start gap-3 p-4 rounded-lg border bg-bg-raised border-border-strong',
              'shadow-[var(--shadow-lg)] animate-[celebrate-spring_400ms_var(--ease-spring)]',
              'backdrop-blur-md',
            )}
          >
            <Icon size={18} className={cn('shrink-0 mt-0.5', KIND_COLOR[t.kind])} />
            <div className="flex-1 min-w-0">
              <div className="text-fg-1 text-[13px] font-semibold">{t.title}</div>
              {t.body && (
                <div className="text-fg-2 text-[12px] mt-0.5 leading-snug">{t.body}</div>
              )}
            </div>
            <button
              onClick={() => dismiss(t.id)}
              aria-label="Dismiss"
              className="text-fg-3 hover:text-fg-1 transition-colors"
            >
              <X size={14} />
            </button>
          </div>
        );
      })}
    </div>
  );
}
