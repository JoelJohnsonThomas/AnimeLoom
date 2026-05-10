import { useQuery } from '@tanstack/react-query';
import { getJob } from '../api';
import { isTerminalStatus } from '../types';

export const JOB_KEYS = {
  detail: (id: string) => ['job', id] as const,
};

/**
 * Polls /job/{id} every 1.5s while the job is pending/running.
 * Stops automatically on completed/failed.
 */
export function useJob(jobId: string | undefined) {
  return useQuery({
    queryKey: jobId ? JOB_KEYS.detail(jobId) : ['job', 'none'],
    queryFn: () => getJob(jobId!),
    enabled: Boolean(jobId),
    refetchInterval: (q) => {
      const status = q.state.data?.status;
      if (!status) return 1500;
      return isTerminalStatus(status) ? false : 1500;
    },
  });
}
