/**
 * Decorative sakura petal overlay. Pure SVG, CSS animation only.
 * Use sparingly — Dashboard hero, completion celebration.
 */
interface SakuraOverlayProps {
  count?: number;
  className?: string;
}

const PETAL_PATH =
  'M12 2 C 14 6, 14 10, 12 12 C 10 10, 10 6, 12 2 Z';

export function SakuraOverlay({ count = 6, className }: SakuraOverlayProps) {
  return (
    <div
      aria-hidden
      className={className}
      style={{ position: 'absolute', inset: 0, pointerEvents: 'none', overflow: 'hidden' }}
    >
      {Array.from({ length: count }).map((_, i) => {
        const left = (i / count) * 100 + Math.random() * 8;
        const delay = -Math.random() * 30;
        const duration = 22 + Math.random() * 14;
        const scale = 0.6 + Math.random() * 0.8;
        return (
          <svg
            key={i}
            viewBox="0 0 24 24"
            width={18 * scale}
            height={18 * scale}
            style={{
              position: 'absolute',
              left: `${left}%`,
              top: 0,
              animation: `sakura-drift ${duration}s linear ${delay}s infinite`,
              opacity: 0,
              filter: 'drop-shadow(0 0 6px rgba(255,107,157,0.4))',
            }}
          >
            <path d={PETAL_PATH} fill="var(--color-pink)" opacity="0.7" />
          </svg>
        );
      })}
    </div>
  );
}
