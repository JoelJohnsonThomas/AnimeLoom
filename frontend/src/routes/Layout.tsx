import { Outlet } from 'react-router-dom';
import { Sidebar } from '@/components/Sidebar';
import { Toaster } from '@/routes/Toaster';

export function Layout() {
  return (
    <div className="min-h-screen flex bg-bg-base">
      <Sidebar />
      <main className="flex-1 min-w-0 overflow-x-hidden">
        <Outlet />
      </main>
      <Toaster />
    </div>
  );
}
