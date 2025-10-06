import { describe, it, expect, vi, beforeEach } from 'vitest';
import { runWithTools } from '../runWithTools';
import type { Ai, RoleScopedChatInput, AiTextGenerationOutput } from '@cloudflare/workers-types';
import type { AiTextGenerationToolInputWithFunction, ProgressEvent } from '../types';

const MODEL = '@cf/meta/llama-3.3-70b-instruct-fp8-fast' as const;

function createMockAi(responses: any[]): Ai {
	let callCount = 0;
	return {
		run: vi.fn(async () => responses[callCount++]),
	} as unknown as Ai;
}

function createStreamingResponse(chunks: string[]): ReadableStream {
	return new ReadableStream({
		async start(controller) {
			for (const chunk of chunks) {
				const sseData = `data: ${JSON.stringify({ response: chunk })}\n\n`;
				controller.enqueue(new TextEncoder().encode(sseData));
			}
			controller.close();
		},
	});
}

describe('runWithTools', () => {
	const testTool: AiTextGenerationToolInputWithFunction = {
		name: 'get_weather',
		description: 'Get the weather for a location',
		parameters: {
			type: 'object' as const,
			properties: {
				location: { type: 'string' as const, description: 'The location' },
			},
			required: ['location'],
		},
		function: async (args: unknown) => {
			const { location } = args as { location: string };
			return JSON.stringify({ temperature: 72, location });
		},
	};

	const messages: RoleScopedChatInput[] = [
		{ role: 'user', content: 'What is the weather in San Francisco?' },
	];

	const runTest = (ai: Ai, config: any = {}) =>
		runWithTools(ai, MODEL, { messages, tools: [testTool] }, config);

	describe('basic functionality', () => {
		it('should execute tool and return final response', async () => {
			const mockAi = createMockAi([
				{ tool_calls: [{ name: 'get_weather', arguments: { location: 'San Francisco' } }] },
				{ response: 'The weather in San Francisco is 72°F' },
			]);

			const result = await runTest(mockAi);

			expect(result.response).toBe('The weather in San Francisco is 72°F');
			expect(mockAi.run).toHaveBeenCalledTimes(2);
		});

		it('should handle response without tool calls', async () => {
			const mockAi = createMockAi([
				{ tool_calls: [] },
				{ response: 'I cannot help with that' },
			]);

			const result = await runTest(mockAi);

			expect(result.response).toBe('I cannot help with that');
			expect(mockAi.run).toHaveBeenCalledTimes(2);
		});

		it('should handle max_tokens parameter', async () => {
			const mockAi = createMockAi([{ response: 'Short response' }]);

			await runWithTools(mockAi, MODEL, { messages, tools: [testTool], max_tokens: 100 });

			expect(mockAi.run).toHaveBeenCalledWith(MODEL, expect.objectContaining({ max_tokens: 100 }));
		});
	});

	describe('progress events', () => {
		it('should emit progress events for tool execution', async () => {
			const mockAi = createMockAi([
				{ tool_calls: [{ name: 'get_weather', arguments: { location: 'SF' } }] },
				{ response: 'Done' },
			]);

			const progressEvents: ProgressEvent[] = [];
			await runTest(mockAi, {
				onProgress: async (event) => progressEvents.push(event),
			});

			expect(progressEvents).toHaveLength(4);
			expect(progressEvents[0]).toMatchObject({
				stage: 'tool_start',
				tool: 'get_weather',
				arguments: { location: 'SF' },
			});
			expect(progressEvents[1]).toMatchObject({
				stage: 'tool_complete',
				tool: 'get_weather',
			});
			expect(progressEvents[2]).toMatchObject({
				stage: 'generating_response',
			});
		expect(progressEvents[3]).toMatchObject({
			stage: 'complete',
		});
		});

		it('should handle errors in onProgress callback', async () => {
			const mockAi = createMockAi([
				{ tool_calls: [{ name: 'get_weather', arguments: { location: 'SF' } }] },
				{ response: 'Done' },
			]);

			const result = await runTest(mockAi, {
				onProgress: async () => {
					throw new Error('Callback error');
				},
			});

			expect(result.response).toBe('Done');
		});

	it('should emit tool_error progress event on tool failure', async () => {
		const errorTool: AiTextGenerationToolInputWithFunction = {
			name: 'failing_tool',
			description: 'A tool that fails',
			parameters: { type: 'object' as const, properties: {}, required: [] },
			function: async () => {
				throw new Error('Tool execution failed');
			},
		};

		const mockAi = createMockAi([
			{ tool_calls: [{ name: 'failing_tool', arguments: {} }] },
			{ response: 'Handled error' },
		]);

		const progressEvents: ProgressEvent[] = [];
		await runWithTools(
			mockAi,
			MODEL,
			{ messages, tools: [errorTool] },
			{
				onProgress: async (event) => progressEvents.push(event),
			}
		);

		const toolErrorEvent = progressEvents.find((e) => e.stage === 'tool_error');
		expect(toolErrorEvent).toBeDefined();
		if (toolErrorEvent && toolErrorEvent.stage === 'tool_error') {
			expect(toolErrorEvent.tool).toBe('failing_tool');
			expect(toolErrorEvent.error).toBe('Tool execution failed');
		}
	});

	it('should emit complete progress event for both streaming and non-streaming', async () => {
		const mockAi1 = createMockAi([
			{ tool_calls: [] },
			{ response: 'Non-streaming complete' },
		]);

		const progressEvents1: ProgressEvent[] = [];
		await runTest(mockAi1, {
			onProgress: async (event) => progressEvents1.push(event),
		});

		const completeEvent1 = progressEvents1.find((e) => e.stage === 'complete');
		expect(completeEvent1).toBeDefined();

		const mockAi2 = createMockAi([
			{ tool_calls: [] },
			createStreamingResponse(['test']),
		]);

		const progressEvents2: ProgressEvent[] = [];
		await runTest(mockAi2, {
			streamFinalResponse: true,
			onChunk: async () => {},
			onProgress: async (event) => progressEvents2.push(event),
		});

		const completeEvent2 = progressEvents2.find((e) => e.stage === 'complete');
		expect(completeEvent2).toBeDefined();
	});
	});

	describe('streaming', () => {
		it('should stream response with onChunk callback', async () => {
			const chunks = ['Hello', ' ', 'world', '!'];
			const mockAi = createMockAi([
				{ tool_calls: [{ name: 'get_weather', arguments: { location: 'SF' } }] },
				createStreamingResponse(chunks),
			]);

			const receivedChunks: Uint8Array[] = [];
			await runTest(mockAi, {
				streamFinalResponse: true,
				onChunk: async (chunk) => receivedChunks.push(chunk),
			});

			expect(receivedChunks.length).toBeGreaterThan(0);
			const fullText = new TextDecoder().decode(
				new Uint8Array(receivedChunks.flatMap((c) => Array.from(c)))
			);
			expect(fullText).toContain('Hello');
			expect(fullText).toContain('world');
		});

		it('should handle errors in onChunk callback', async () => {
			const mockAi = createMockAi([
				{ tool_calls: [{ name: 'get_weather', arguments: { location: 'SF' } }] },
				createStreamingResponse(['test']),
			]);

			const result = await runTest(mockAi, {
				streamFinalResponse: true,
				onChunk: async () => {
					throw new Error('Chunk error');
				},
			});

			expect(result.response).toBe('');
		});

		it('should batch chunks when streaming', async () => {
			const mockAi = createMockAi([
				{ tool_calls: [{ name: 'get_weather', arguments: { location: 'SF' } }] },
				createStreamingResponse(Array(100).fill(null).map((_, i) => `${i}`)),
			]);

			let chunkCallCount = 0;
			await runTest(mockAi, {
				streamFinalResponse: true,
				onChunk: async () => chunkCallCount++,
			});

			expect(chunkCallCount).toBeLessThan(100);
		});
	});

	describe('abort signal', () => {
		it('should stop streaming when signal is aborted', async () => {
			const controller = new AbortController();
			const longStream = new ReadableStream({
				async start(ctrl) {
					for (let i = 0; i < 1000; i++) {
						ctrl.enqueue(
							new TextEncoder().encode(`data: ${JSON.stringify({ response: `${i}` })}\n\n`)
						);
						await new Promise((resolve) => setTimeout(resolve, 1));
					}
					ctrl.close();
				},
			});

			const mockAi = createMockAi([
				{ tool_calls: [{ name: 'get_weather', arguments: { location: 'SF' } }] },
				longStream,
			]);

			setTimeout(() => controller.abort(), 50);

			let chunkCount = 0;
			const result = await runTest(mockAi, {
				streamFinalResponse: true,
				signal: controller.signal,
				onChunk: async () => chunkCount++,
			});

			expect(chunkCount).toBeLessThan(1000);
			expect(result.response).toBe('');
		});
	});

	describe('tool errors', () => {
		it('should handle tool execution errors', async () => {
			const errorTool: AiTextGenerationToolInputWithFunction = {
				name: 'failing_tool',
				description: 'A tool that fails',
				parameters: { type: 'object' as const, properties: {}, required: [] },
				function: async () => {
					throw new Error('Tool error');
				},
			};

			const mockAi = createMockAi([
				{ tool_calls: [{ name: 'failing_tool', arguments: {} }] },
				{ response: 'Handled error' },
			]);

			const result = await runWithTools(mockAi, MODEL, { messages, tools: [errorTool] });

			expect(result.response).toBe('Handled error');
		});

		it('should handle missing tool', async () => {
			const mockAi = createMockAi([
				{ tool_calls: [{ name: 'nonexistent_tool', arguments: {} }] },
				{ response: 'Recovered' },
			]);

			const result = await runTest(mockAi);

			expect(result.response).toBe('Recovered');
		});
	});

	describe('parallel tool execution', () => {
		it('should execute multiple tools in parallel with correct ordering', async () => {
			const fastTool: AiTextGenerationToolInputWithFunction = {
				name: 'fast_tool',
				description: 'Fast tool',
				parameters: { type: 'object' as const, properties: {}, required: [] },
				function: async () => {
					await new Promise((resolve) => setTimeout(resolve, 10));
					return JSON.stringify({ result: 'fast' });
				},
			};

			const slowTool: AiTextGenerationToolInputWithFunction = {
				name: 'slow_tool',
				description: 'Slow tool',
				parameters: { type: 'object' as const, properties: {}, required: [] },
				function: async () => {
					await new Promise((resolve) => setTimeout(resolve, 50));
					return JSON.stringify({ result: 'slow' });
				},
			};

			const mockAi = createMockAi([
				{
					tool_calls: [
						{ name: 'slow_tool', arguments: {} },
						{ name: 'fast_tool', arguments: {} },
					],
				},
				{ response: 'Both tools completed' },
			]);

			const progressEvents: ProgressEvent[] = [];
			const result = await runWithTools(
				mockAi,
				MODEL,
				{ messages, tools: [fastTool, slowTool] },
				{
					onProgress: async (event) => progressEvents.push(event),
				}
			);

			expect(result.response).toBe('Both tools completed');
			expect(mockAi.run).toHaveBeenCalledTimes(2);

			const toolStartEvents = progressEvents.filter((e) => e.stage === 'tool_start');
			const toolCompleteEvents = progressEvents.filter((e) => e.stage === 'tool_complete');

			expect(toolStartEvents).toHaveLength(2);
			expect(toolCompleteEvents).toHaveLength(2);

			const fastCompleteIndex = progressEvents.findIndex(
				(e) => e.stage === 'tool_complete' && e.tool === 'fast_tool'
			);
			const slowCompleteIndex = progressEvents.findIndex(
				(e) => e.stage === 'tool_complete' && e.tool === 'slow_tool'
			);

			expect(fastCompleteIndex).toBeLessThan(slowCompleteIndex);
		});
	});

	describe('recursive tool runs', () => {
		it('should respect maxRecursiveToolRuns', async () => {
			const mockAi = createMockAi([
				{ tool_calls: [{ name: 'get_weather', arguments: { location: 'SF' } }] },
				{ tool_calls: [{ name: 'get_weather', arguments: { location: 'LA' } }] },
				{ response: 'Done' },
			]);

			await runTest(mockAi, { maxRecursiveToolRuns: 1 });

			expect(mockAi.run).toHaveBeenCalledTimes(3);
		});

		it('should stop at max recursive runs', async () => {
			const mockAi = createMockAi([
				{ tool_calls: [{ name: 'get_weather', arguments: { location: 'SF' } }] },
				{ tool_calls: [{ name: 'get_weather', arguments: { location: 'LA' } }] },
				{ response: 'Final' },
			]);

			await runTest(mockAi, { maxRecursiveToolRuns: 0 });

			expect(mockAi.run).toHaveBeenCalledTimes(2);
		});
	});

	describe('type safety', () => {
		it('should enforce discriminated union types', () => {
			const handleProgress = (event: ProgressEvent) => {
				if (event.stage === 'tool_start') {
					expect(event.tool).toBeDefined();
					expect(event.arguments).toBeDefined();
				} else if (event.stage === 'tool_complete') {
					expect(event.tool).toBeDefined();
					expect(event.result).toBeDefined();
				} else if (event.stage === 'generating_response') {
					expect(event).not.toHaveProperty('tool');
				}
			};

			const toolStart: ProgressEvent = {
				stage: 'tool_start',
				tool: 'test',
				arguments: {},
			};
			handleProgress(toolStart);

			const toolComplete: ProgressEvent = {
				stage: 'tool_complete',
				tool: 'test',
				result: {},
			};
			handleProgress(toolComplete);

			const generating: ProgressEvent = {
				stage: 'generating_response',
			};
			handleProgress(generating);
		});

		it('should return AiTextGenerationOutput for non-streaming', async () => {
			const mockAi = createMockAi([
				{ tool_calls: [] },
				{ response: 'Non-streaming' },
			]);

			const result = await runTest(mockAi);

			expect(result).toHaveProperty('response');
			expect(result.response).toBe('Non-streaming');
		});

		it('should return AiTextGenerationOutput for explicit streamFinalResponse: false', async () => {
			const mockAi = createMockAi([
				{ tool_calls: [] },
				{ response: 'Explicit false' },
			]);

			const result = await runTest(mockAi, { streamFinalResponse: false });

			expect(result).toHaveProperty('response');
			expect(result.response).toBe('Explicit false');
		});

		it('should return empty response for streaming with onChunk', async () => {
			const mockAi = createMockAi([
				{ tool_calls: [] },
				createStreamingResponse(['test']),
			]);

			const result = await runTest(mockAi, {
				streamFinalResponse: true,
				onChunk: async () => {},
			});

			expect(result).toHaveProperty('response');
			expect(result.response).toBe('');
		});
	});
});
