import type { ReactNode } from 'react';
import { cn } from '../utils';

export type BadgeStatus =
  | 'completed'
  | 'running'
  | 'pending'
  | 'failed'
  | 'training'
  | 'draft'
  | 'high-quality'
  | 'info'
  | 'neutral';

interface BadgeProps {
  status?: BadgeStatus;
  children: ReactNode;
  className?: string;
}

const PALETTE: Record<BadgeStatus, string> = {
  completed: 'bg-[var(--green-bg)] text-green border border-green/20',
  running: 'bg-[var(--info-bg)] text-blue border border-blue/20',
  pending: 'bg-[var(--yellow-bg)] text-yellow border border-yellow/20',
  failed: 'bg-[var(--red-bg)] text-red border border-red/20',
  training: 'bg-[var(--info-bg)] text-blue border border-blue/20',
  draft: 'bg-bg-raised text-fg-2 border border-border-subtle',
  'high-quality': 'bg-[var(--gold-glow)] text-gold border border-gold/20',
  info: 'bg-[var(--info-bg)] text-blue border border-blue/20',
  neutral: 'bg-bg-raised text-fg-2 border border-border-subtle',
};

export function Badge({ status = 'neutral', children, className }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center px-2.5 py-0.5 rounded-full text-[10px] font-semibold uppercase tracking-wider',
        PALETTE[status],
        className,
      )}
    >
      {children}
    </span>
  );
}
