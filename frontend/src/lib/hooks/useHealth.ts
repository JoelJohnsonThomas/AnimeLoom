import { useQuery } from '@tanstack/react-query';
import { getHealth } from '../api';

export function useHealth() {
  return useQuery({
    queryKey: ['health'],
    queryFn: getHealth,
    staleTime: 60_000,
  });
}
