import { Logger } from "./logger";
import { validateArgsWithZod } from "./utils";
import {
	Ai,
	AiTextGenerationInput,
	AiTextGenerationOutput,
	RoleScopedChatInput,
} from "@cloudflare/workers-types";
import { AiTextGenerationToolInputWithFunction, ModelName, ProgressEvent } from "./types";

export interface RunWithToolsInput {
	messages: RoleScopedChatInput[];
	tools: AiTextGenerationToolInputWithFunction[];
	max_tokens?: number;
}

export interface RunWithToolsConfig {
	streamFinalResponse?: boolean;
	maxRecursiveToolRuns?: number;
	strictValidation?: boolean;
	verbose?: boolean;
	onProgress?: (event: ProgressEvent) => void | Promise<void>;
	onChunk?: (chunk: Uint8Array) => void | Promise<void>;
	signal?: AbortSignal;
	trimFunction?: (
		tools: AiTextGenerationToolInputWithFunction[],
		ai: Ai,
		model: ModelName,
		messages: RoleScopedChatInput[],
	) => Promise<AiTextGenerationToolInputWithFunction[]>;
}

/**
 * Runs a set of tools on a given input and returns the final response in the same format as the AI.run call.
 *
 * @param {Ai} ai - The AI instance to use for the run.
 * @param {ModelName} model - The function calling model to use for the run. We recommend using `@hf/nousresearch/hermes-2-pro-mistral-7b`, `llama-3` or equivalent model that's suited for function calling.
 * @param {RunWithToolsInput} input - The input for the runWithTools call.
 * @param {RunWithToolsConfig} config - Configuration options for the runWithTools call.
 *
 * @returns {Promise<AiTextGenerationOutput>} The final response when not streaming.
 */
export async function runWithTools(
	ai: Ai,
	model: ModelName,
	input: RunWithToolsInput,
	config?: RunWithToolsConfig & { streamFinalResponse?: false }
): Promise<AiTextGenerationOutput>;

/**
 * Runs a set of tools on a given input and returns a streaming response.
 *
 * @param {Ai} ai - The AI instance to use for the run.
 * @param {ModelName} model - The function calling model to use for the run.
 * @param {RunWithToolsInput} input - The input for the runWithTools call.
 * @param {RunWithToolsConfig} config - Configuration options with streaming enabled.
 *
 * @returns {Promise<AiTextGenerationOutput>} Empty response object when streaming (actual content via onChunk callback).
 */
export async function runWithTools(
	ai: Ai,
	model: ModelName,
	input: RunWithToolsInput,
	config: RunWithToolsConfig & { streamFinalResponse: true }
): Promise<AiTextGenerationOutput>;

export async function runWithTools(
	ai: Ai,
	model: ModelName,
	input: RunWithToolsInput,
	config: RunWithToolsConfig = {},
): Promise<AiTextGenerationOutput> {
	// Destructure config with default values
	const {
		streamFinalResponse = false,
		maxRecursiveToolRuns = 0,
		verbose = false,
		trimFunction = async (
			tools: AiTextGenerationToolInputWithFunction[],
			ai: Ai,
			model: ModelName,
			messages: RoleScopedChatInput[],
		) => tools as AiTextGenerationToolInputWithFunction[],
		strictValidation = false,
		onProgress,
		onChunk,
		signal,
	} = config;

	// Enable verbose logging if specified in the config
	if (verbose) {
		Logger.enableLogging();
	}

	// Remove functions from the tools for initial processing
	const initialtoolsWithoutFunctions = input.tools.map(
		({ function: _function, ...rest }) => rest,
	);

	// Transform tools to include only the function definitions
	let tools = initialtoolsWithoutFunctions.map((tool) => ({
		type: "function" as const,
		function: { ...tool, function: undefined },
	}));

	let tool_calls: { name: string; arguments: unknown }[] = [];
	let totalCharacters = 0;

	// Creating a copy of the input object to avoid mutating the original object
	const messages = [...input.messages];

	// If trimFunction is enabled, choose the best tools for the task
	if (trimFunction) {
		const chosenTools = await trimFunction(input.tools, ai, model, messages);
		tools = chosenTools.map((tool) => ({
			type: "function",
			function: { ...tool, function: undefined },
		}));
	}

	// Recursive function to process responses and execute tools
	async function runAndProcessToolCall({
		ai,
		model,
		messages,
		streamFinalResponse,
		maxRecursiveToolRuns,
	}: {
		ai: Ai;
		model: ModelName;
		messages: RoleScopedChatInput[];
		streamFinalResponse: boolean;
		maxRecursiveToolRuns: number;
	}): Promise<AiTextGenerationOutput> {
		try {
			Logger.info("Starting AI.run call");
			Logger.info("Messages", JSON.stringify(messages, null, 2));

			Logger.info(`Only using ${input.tools.length} tools`);

			const response = (await ai.run(model, {
				messages: messages,
				stream: false,
				tools: tools,
				...(input.max_tokens !== undefined && { max_tokens: input.max_tokens }),
			})) as {
				response?: string;
				tool_calls?: {
					name: string;
					arguments: unknown;
				}[];
			};

			const chars =
				JSON.stringify(messages).length +
				JSON.stringify(initialtoolsWithoutFunctions).length;
			totalCharacters += chars;
			Logger.info(
				`Number of characters for the first AI.run call: ${totalCharacters}`,
			);

			Logger.info("AI.run call completed", response);

			tool_calls = response.tool_calls?.filter(Boolean) ?? [];

			if (tool_calls.length > 0) {
				messages.push({
					role: "assistant",
					content: JSON.stringify(tool_calls),
				});

				const toolResults = await Promise.all(
					tool_calls.map(async (toolCall, index) => {
						const selectedTool = input.tools.find(
							(tool) => tool.name === toolCall.name,
						);

						if (!selectedTool) {
							Logger.error(
								`Tool ${toolCall.name} not found, maybe AI hallucinated`,
							);
							return { index, toolCall, result: null };
						}

						const fn = selectedTool.function;

						if (fn === undefined || selectedTool.parameters === undefined) {
							Logger.error(
								`Function for tool ${toolCall.name} is undefined`,
							);
							return { index, toolCall, result: null };
						}

						const args = toolCall.arguments;

						if (
							strictValidation &&
							!validateArgsWithZod(
								args,
								selectedTool.parameters.properties as any,
							)
						) {
							Logger.error(
								`Invalid arguments for tool ${selectedTool.name}: ${JSON.stringify(args)}`,
							);
							return { index, toolCall, result: null };
						}

						try {
							Logger.info(
								`Executing tool ${selectedTool.name} with arguments`,
								args,
							);

							if (onProgress) {
								try {
									await onProgress({
										stage: 'tool_start',
										tool: selectedTool.name,
										arguments: args,
										message: `Executing ${selectedTool.name}`,
									});
								} catch (error) {
									Logger.error('onProgress callback failed:', error);
								}
							}

							const result = await fn(args);

							Logger.info(`Tool ${selectedTool.name} execution result`, result);

							if (onProgress) {
								try {
									await onProgress({
										stage: 'tool_complete',
										tool: selectedTool.name,
										result,
										message: `Completed ${selectedTool.name}`,
									});
								} catch (error) {
									Logger.error('onProgress callback failed:', error);
								}
							}

							return { index, toolCall, result, toolName: selectedTool.name };
						} catch (error) {
							Logger.error(`Error executing tool ${selectedTool.name}:`, error);

						if (onProgress) {
							try {
								await onProgress({
									stage: 'tool_error',
									tool: selectedTool.name,
									error: (error as Error).message,
									message: `Failed to execute ${selectedTool.name}`,
								});
							} catch (progressError) {
								Logger.error('onProgress callback failed:', progressError);
							}
						}
							return {
								index,
								toolCall,
								result: null,
								error: (error as Error).message,
								toolName: selectedTool.name,
							};
						}
					})
				);

				for (const { toolCall, result, error, toolName } of toolResults) {
					if (error) {
						messages.push({
							role: "tool",
							content: `Error executing tool ${toolName}: ${error}`,
							name: toolCall.name,
						});
					} else if (result !== null) {
						messages.push({
							role: "tool",
							content: JSON.stringify(result),
							name: toolCall.name,
						});
					}
				}
			}

			// Recursively call the runAndProcessToolCall if maxRecursiveToolRuns is not reached
			if (maxRecursiveToolRuns > 0 && tool_calls.length > 0) {
				maxRecursiveToolRuns--;
				return await runAndProcessToolCall({
					ai,
					model,
					messages,
					streamFinalResponse,
					maxRecursiveToolRuns,
				});
			} else {
				Logger.info(
					"Max recursive tool runs reached, generating final response",
				);

				// Emit progress event: generating final response
				if (onProgress) {
					try {
						await onProgress({
							stage: 'generating_response',
							message: 'Generating response',
						});
					} catch (error) {
						Logger.error('onProgress callback failed:', error);
					}
				}

				const finalResponse = await ai.run(model, {
					messages: messages,
					stream: streamFinalResponse,
					...(input.max_tokens !== undefined && { max_tokens: input.max_tokens }),
				});

				totalCharacters += JSON.stringify(messages).length;
				Logger.info(
					`Number of characters for the final AI.run call: ${JSON.stringify(messages).length}`,
				);

				// If streaming with onChunk callback, consume the stream here and forward raw bytes
				if (streamFinalResponse && onChunk) {
					Logger.info("Consuming stream with onChunk callback - forwarding raw bytes");

					const reader = (finalResponse as ReadableStream).getReader();

					// Chunk batching for better performance
					const BATCH_SIZE = 4096; // 4KB
					const BATCH_TIMEOUT_MS = 10; // 10ms max delay
					let buffer: Uint8Array[] = [];
					let bufferSize = 0;
					let timeoutId: NodeJS.Timeout | null = null;

					const flushBuffer = async () => {
						if (buffer.length === 0) return;

						const combined = new Uint8Array(bufferSize);
						let offset = 0;
						for (const chunk of buffer) {
							combined.set(chunk, offset);
							offset += chunk.byteLength;
						}

						buffer = [];
						bufferSize = 0;
						if (timeoutId) {
							clearTimeout(timeoutId);
							timeoutId = null;
						}

						try {
							await onChunk(combined);
						} catch (error) {
							Logger.error('onChunk callback failed:', error);
						}
					};

					while (true) {
						if (signal?.aborted) {
							Logger.info('Stream aborted by signal');
							await flushBuffer();
							break;
						}

						const { done, value } = await reader.read();
						if (done) {
							await flushBuffer();
							break;
						}

						buffer.push(value);
						bufferSize += value.byteLength;

						if (bufferSize >= BATCH_SIZE) {
							await flushBuffer();
						} else if (!timeoutId) {
							timeoutId = setTimeout(() => flushBuffer(), BATCH_TIMEOUT_MS) as unknown as NodeJS.Timeout;
						}
					}

					Logger.info(`Total number of characters: ${totalCharacters}`);

				if (onProgress) {
					try {
						await onProgress({
							stage: 'complete',
							message: 'Stream complete',
						});
					} catch (error) {
						Logger.error('onProgress callback failed:', error);
					}
				}
					// Return empty response since consumer handles the stream
					return { response: '' } as AiTextGenerationOutput;
				}

				Logger.info(`Total number of characters: ${totalCharacters}`);

				if (onProgress) {
					try {
						await onProgress({
							stage: 'complete',
							message: 'Response complete',
						});
					} catch (error) {
						Logger.error('onProgress callback failed:', error);
					}
				}
				return finalResponse as AiTextGenerationOutput;
			}
		} catch (error) {
			Logger.error("Error in runAndProcessToolCall:", error);
			throw new Error(
				`Error in runAndProcessToolCall: ${(error as Error).message}`,
			);
		}
	}

	try {
		Logger.info("Starting runWithTools process");
		const result = await runAndProcessToolCall({
			ai,
			model,
			messages,
			streamFinalResponse,
			maxRecursiveToolRuns,
		});
		Logger.info("runWithTools process completed");
		return result;
	} catch (error) {
		Logger.error("Error in runWithTools:", error);
		throw new Error(`Error in runWithTools: ${(error as Error).message}`);
	}
};
