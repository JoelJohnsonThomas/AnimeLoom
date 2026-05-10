import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { RouterProvider, createBrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

import './styles/tokens.css';
import './styles/base.css';

import { Layout } from '@/routes/Layout';
import { Dashboard } from '@/routes/Dashboard';
import { Characters } from '@/routes/Characters';
import { ScriptEditor } from '@/routes/ScriptEditor';
import { Generate } from '@/routes/Generate';
import { Results } from '@/routes/Results';
import { Settings } from '@/routes/Settings';
import { Docs } from '@/routes/Docs';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 30_000,
    },
  },
});

const router = createBrowserRouter(
  [
    {
      path: '/',
      element: <Layout />,
      children: [
        { index: true, element: <Dashboard /> },
        { path: 'characters', element: <Characters /> },
        { path: 'script', element: <ScriptEditor /> },
        { path: 'generate', element: <Generate /> },
        { path: 'generate/:jobId', element: <Generate /> },
        { path: 'results', element: <Results /> },
        { path: 'settings', element: <Settings /> },
        { path: 'docs', element: <Docs /> },
      ],
    },
  ],
  { basename: import.meta.env.VITE_BASE?.replace(/\/$/, '') || '/' },
);

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </StrictMode>,
);
