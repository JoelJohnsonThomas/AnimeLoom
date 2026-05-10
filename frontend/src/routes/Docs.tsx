import { ExternalLink, BookOpen } from 'lucide-react';
import { Card } from '@/components/ui/Card';

export function Docs() {
  const apiTarget = import.meta.env.VITE_API_URL ?? '/api';
  // Default-dev FastAPI runs on :8080; build into the link so users can click out.
  const swaggerUrl = apiTarget.startsWith('http') ? `${apiTarget}/docs` : 'http://localhost:8080/docs';

  return (
    <div className="p-8 flex flex-col gap-6 max-w-[1100px]">
      <header className="flex items-end justify-between gap-4">
        <div>
          <h1 className="text-[28px]">Docs</h1>
          <p className="text-fg-hint text-[13px] mt-1">FastAPI Swagger reference for the AnimeLoom backend.</p>
        </div>
        <a
          href={swaggerUrl}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-2 text-blue hover:text-fg-1 text-[13px]"
        >
          Open in new tab <ExternalLink size={13} />
        </a>
      </header>

      <Card>
        <div className="flex items-start gap-3">
          <BookOpen size={18} className="text-pink mt-0.5" strokeWidth={1.5} />
          <div className="flex flex-col gap-3 text-[13px]">
            <div className="text-fg-2">
              The Swagger UI is served by FastAPI at <code className="font-mono text-fg-code">/docs</code>.
              If the embedded iframe below shows "refused to connect", open the link in a new tab.
            </div>
            <div className="grid grid-cols-2 gap-2 text-[12px] text-fg-hint">
              <span>Auth</span>
              <span className="font-mono text-fg-2">none (CORS *)</span>
              <span>Streaming</span>
              <span className="font-mono text-fg-2">REST polling, no WebSocket</span>
              <span>Endpoints</span>
              <span className="font-mono text-fg-2">10 (characters · generation · job)</span>
            </div>
          </div>
        </div>
      </Card>

      <Card padded={false} className="overflow-hidden">
        <iframe
          src={swaggerUrl}
          title="FastAPI Swagger"
          className="w-full"
          style={{ height: '70vh', border: 'none', background: 'white' }}
        />
      </Card>
    </div>
  );
}
