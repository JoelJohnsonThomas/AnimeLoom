# AnimeLoom Web Studio

Production Vite + React + TypeScript frontend for AnimeLoom. Wires the FastAPI backend (`api/app.py`) into a branded Web Studio: characters, scripts, generation jobs, results.

## Stack

| Choice | Why |
|---|---|
| Vite 6 + React 18 + TypeScript | Fast HMR, modern, no Next.js overkill |
| Tailwind CSS v4 | CSS-first config; tokens map 1:1 to design-system CSS vars |
| React Router 6 | 7-screen SPA navigation |
| TanStack Query 5 | Polls the async-job API cleanly (no WebSocket needed) |
| Lucide React | Same icon set the original JSX used |
| Zustand | Tiny store for ephemeral state (active job, toasts) |

## Setup

```bash
cd frontend
npm install
```

## Run modes

### 1 — Standalone dev (recommended for frontend work)

```bash
npm run dev
```

Opens http://localhost:5173. API calls go to `/api/*` and are proxied to FastAPI on `:8080` via `vite.config.ts`. Start the backend separately:

```bash
# in another terminal, from the repo root
python main.py --api
```

### 2 — Embedded in FastAPI (recommended for deployment)

```bash
npm run build:embedded
# back at repo root
python main.py --api
```

Visit http://localhost:8080/ui/. FastAPI serves the built static files from `frontend/dist/` at `/ui/`. `api/app.py` mounts this automatically if the directory exists.

### 3 — Standalone production build

```bash
npm run build
```

Outputs to `frontend/dist/`. Deploy anywhere (Vercel, Netlify, S3+CloudFront). Set `VITE_API_URL` at build time to the absolute backend URL.

## Architecture

```
src/
├── main.tsx                 # Router + QueryClient root
├── styles/
│   ├── tokens.css           # Tailwind v4 @theme — single source of truth
│   └── base.css             # Focus-visible ring + scrollbar + animations
├── routes/                  # 7 screens + Layout + Toaster
├── components/
│   ├── ui/                  # Button, Card, Input, Badge, Progress, Modal
│   └── brand/               # LogoMark, SakuraOverlay
└── lib/
    ├── types.ts             # TS mirror of api/schemas/models.py
    ├── api.ts               # Typed fetch wrappers
    ├── hooks/               # TanStack Query hooks
    └── store.ts             # Zustand
```

## Design tokens

Tokens live **only** in `src/styles/tokens.css` (`@theme` block). Components use Tailwind classes that read those tokens (e.g. `bg-pink`, `text-fg-1`, `border-border-subtle`). No hardcoded hex anywhere in the JSX.

`AnimeLoom Design System/colors_and_type.css` is the historical artifact and design-spec reference. Real wiring lives here.

### A11y

- **`:focus-visible` ring** — pink outline + soft glow on every interactive element. Defined in `base.css`.
- **`--fg-3` is decorative only** — timestamps/separators. Body text uses `--fg-2` or `--fg-hint` (both ≥ WCAG AA on `--bg-base`).

## Brand

Anime/pink direction:

- Gradient pink→blue logo mark with glow shadow.
- Pink-glow CTAs (`Button` with `glow` prop).
- Sakura petal overlay on the Dashboard hero (`SakuraOverlay`).
- Spring animation on toast/modal entry (`celebrate-spring` keyframes).
- Voice: technically precise, anime-literate, no emoji in copy.

## Backend contract

- All paths under `/api/*` proxy to FastAPI in dev. In embedded mode they're same-origin under `/`.
- No auth (FastAPI CORS is `*`). Add headers in `lib/api.ts` if/when auth lands.
- Long jobs return a `JobStatus` immediately with a `job_id`; the UI polls `/job/{id}` every 1.5 s via `useJob`.

## Scripts

| Script | What |
|---|---|
| `npm run dev` | Vite dev server, port 5173, proxies `/api` to :8080 |
| `npm run build` | Standalone production build → `dist/` |
| `npm run build:embedded` | Build with `base=/ui/` for FastAPI static mount |
| `npm run preview` | Preview the production build locally |
| `npm run typecheck` | `tsc --noEmit` across the project |

## Env

| Var | Default | Used in |
|---|---|---|
| `VITE_API_URL` | `/api` | `lib/api.ts` base URL |
| `VITE_API_TARGET` | `http://localhost:8080` | dev proxy target |
| `VITE_BASE` | `/` | router basename + Vite `base` |
