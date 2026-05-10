import { cn } from '../utils';

interface ProgressProps {
  /** 0..1 (preferred) or 0..100 — auto-detected */
  value: number;
  label?: string;
  className?: string;
  showPercent?: boolean;
}

export function Progress({ value, label, className, showPercent = true }: ProgressProps) {
  const pct = value > 1 ? Math.max(0, Math.min(100, value)) : Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className={cn('flex flex-col gap-2', className)}>
      {(label || showPercent) && (
        <div className="flex justify-between items-center text-[12px]">
          {label && <span className="text-fg-2">{label}</span>}
          {showPercent && (
            <span className="text-fg-1 font-semibold font-mono">{pct.toFixed(0)}%</span>
          )}
        </div>
      )}
      <div className="h-2 bg-bg-deep rounded-full overflow-hidden border border-border-faint">
        <div
          className="h-full rounded-full transition-[width] duration-500"
          style={{
            width: `${pct}%`,
            background:
              'linear-gradient(90deg, var(--color-pink) 0%, var(--color-blue) 100%)',
            boxShadow: '0 0 12px var(--pink-glow)',
          }}
        />
      </div>
    </div>
  );
}
