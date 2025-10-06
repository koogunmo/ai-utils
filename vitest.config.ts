import { defineWorkersConfig } from '@cloudflare/vitest-pool-workers/config';

export default defineWorkersConfig({
	test: {
		globals: true,
		pool: '@cloudflare/vitest-pool-workers',
		poolOptions: {
			workers: {
				miniflare: {
					compatibilityDate: '2025-10-01',
					compatibilityFlags: ['nodejs_compat'],
				},
			},
		},
	},
});
