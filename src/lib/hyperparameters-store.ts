import { create } from "zustand";
import localForage from "localforage";
import { persist, createJSONStorage } from "zustand/middleware";

export type LayerType = "Dense" | "Conv1D" | "Conv2D" | "Flatten";
export type BuiltinActivationType = "ReLU" | "Sigmoid" | "Tanh";
export type ActivationType = BuiltinActivationType | "Custom" | string; // Allow any string for custom function names
export type OptimizerType = "SGD" | "Adam" | "RMSprop" | "Adagrad";
export type LearningRateScheduleType = "Step" | "Exponential" | "Custom";
export type BatchSizeOption = 16 | 32 | 64 | 128 | "Custom";

export interface CustomActivationFunction {
	name: string;
	code: string;
}

export interface Layer {
	id: string;
	type: LayerType;
	units?: number;
	filters?: number;
	kernelSize?: number;
	activation?: ActivationType;
	customActivation?: string; // Kept for backward compatibility
	customActivationFunctions?: CustomActivationFunction[];
	selectedCustomFunction?: string; // Name of the currently selected custom function
	inputSize?: number;
	outputSize?: number;
}

export interface HyperparametersState {
	layers: Layer[];
	learningRate: number;
	optimizer: OptimizerType;
	batchSize: BatchSizeOption;
	customBatchSize?: number;
	epochs: number;
	dropoutRate: number;
	weightDecay: number;
	gradientClippingThreshold: number;
	learningRateSchedule: LearningRateScheduleType;
	useEarlyStopping: boolean;
	useModelCheckpoint: boolean;
	customScheduleCallback?: string;
	totalParams: number;
	estimatedMemory: number;
	defaultCustomFunction?: string; // Name of the default custom function to use for new layers
	globalCustomFunctions: CustomActivationFunction[]; // Global list of custom functions
}

export interface HyperparametersStore extends HyperparametersState {
	addLayer: (layer: Omit<Layer, "id">) => void;
	updateLayer: (id: string, layer: Partial<Layer>) => void;
	removeLayer: (id: string) => void;
	moveLayerUp: (id: string) => void;
	moveLayerDown: (id: string) => void;
	setLearningRate: (rate: number) => void;
	setOptimizer: (optimizer: OptimizerType) => void;
	setBatchSize: (size: BatchSizeOption, customSize?: number) => void;
	setEpochs: (epochs: number) => void;
	setDropoutRate: (rate: number) => void;
	setDefaultCustomFunction: (name: string) => void;
	addGlobalCustomFunction: (functionName: string, code: string) => void;
	removeGlobalCustomFunction: (functionName: string) => void;
	updateGlobalCustomFunction: (functionName: string, code: string) => void;
	setWeightDecay: (decay: number) => void;
	setGradientClippingThreshold: (threshold: number) => void;
	setLearningRateSchedule: (schedule: LearningRateScheduleType) => void;
	setCustomScheduleCallback: (callback: string) => void;
	toggleEarlyStopping: () => void;
	toggleModelCheckpoint: () => void;
	updateTotals: () => void;
	resetHyperparameters: () => void;
}

// Default settings and initial values
const DEFAULT_SETTINGS: HyperparametersState = {
	layers: [
		{
			id: "layer-1",
			type: "Dense",
			units: 128,
			activation: "ReLU",
			customActivationFunctions: [
				{
					name: "LeakyReLU",
					code: "function LeakyReLU(x) {\n  return x > 0 ? x : 0.01 * x;\n}",
				},
				{
					name: "GELU",
					code: "function GELU(x) {\n  return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));\n}",
				},
			],
		},
		{
			id: "layer-2",
			type: "Dense",
			units: 64,
			activation: "ReLU",
		},
		{
			id: "layer-3",
			type: "Dense",
			units: 10,
			activation: "Sigmoid",
		},
	],
	learningRate: 0.001,
	optimizer: "Adam",
	batchSize: 32,
	epochs: 10,
	dropoutRate: 0.2,
	weightDecay: 0.001,
	gradientClippingThreshold: 1.0,
	learningRateSchedule: "Step",
	useEarlyStopping: true,
	useModelCheckpoint: true,
	totalParams: 0,
	estimatedMemory: 0,
	globalCustomFunctions: [
		{
			name: "LeakyReLU",
			code: "function LeakyReLU(x) {\n  return x > 0 ? x : 0.01 * x;\n}",
		},
		{
			name: "GELU",
			code: "function GELU(x) {\n  return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));\n}",
		},
		{
			name: "Swish",
			code: "function Swish(x) {\n  return x / (1 + Math.exp(-x));\n}",
		},
	],
};

// Helper function to estimate parameters and memory
function calculateModelStats(layers: Layer[]): {
	params: number;
	memory: number;
	layersWithSizes: Layer[];
} {
	let prevUnits = 784; // Assuming MNIST input
	let totalParams = 0;

	// Create a deep copy of layers to update with input/output sizes
	const layersWithSizes = JSON.parse(JSON.stringify(layers));

	for (let i = 0; i < layersWithSizes.length; i++) {
		const layer = layersWithSizes[i];

		// Set input size
		layer.inputSize = prevUnits;

		if (layer.type === "Dense" && layer.units) {
			// Parameters = (input_size + 1) * output_size (including bias)
			const params = (prevUnits + 1) * layer.units;
			totalParams += params;
			prevUnits = layer.units;

			// Set output size
			layer.outputSize = layer.units;
		} else if (layer.type === "Conv2D" && layer.filters && layer.kernelSize) {
			// Simplified Conv2D params calculation
			const kernelSize = layer.kernelSize;
			const params = (kernelSize * kernelSize * prevUnits + 1) * layer.filters;
			totalParams += params;
			prevUnits = layer.filters;

			// Set output size
			layer.outputSize = layer.filters;
		} else if (layer.type === "Conv1D" && layer.filters && layer.kernelSize) {
			// Simplified Conv1D params calculation
			const params = (layer.kernelSize * prevUnits + 1) * layer.filters;
			totalParams += params;
			prevUnits = layer.filters;

			// Set output size
			layer.outputSize = layer.filters;
		} else if (layer.type === "Flatten") {
			// Flatten doesn't add parameters
			layer.outputSize = prevUnits;
		}
	}

	// Rough memory estimate (4 bytes per parameter for float32)
	const memory = (totalParams * 4) / (1024 * 1024); // in MB

	return { params: totalParams, memory, layersWithSizes };
}

// Create the Zustand store with persistence
export const useHyperparametersStore = create<HyperparametersStore>()(
	persist(
		(set, get) => ({
			...DEFAULT_SETTINGS,

			addLayer: (layer) => {
				const state = get();
				
				// Prepare the new layer
				let newLayer = {
					...layer,
					id: `layer-${Date.now()}`,
				};
				
				// If layer is using Custom activation and there's a default function, use it
				if (newLayer.activation === "Custom" && state.defaultCustomFunction) {
					// Find the function in any existing layer
					let functionCode: string | undefined;
					
					for (const existingLayer of state.layers) {
						if (existingLayer.customActivationFunctions) {
							const func = existingLayer.customActivationFunctions.find(
								f => f.name === state.defaultCustomFunction
							);
							if (func) {
								functionCode = func.code;
								break;
							}
						}
					}
					
					// If we found the function, apply it to the new layer
					if (functionCode) {
						newLayer = {
							...newLayer,
							selectedCustomFunction: state.defaultCustomFunction,
							customActivation: functionCode,
							customActivationFunctions: [
								{ name: state.defaultCustomFunction, code: functionCode }
							]
						};
					}
				}

				set((state) => {
					const newLayers = [...state.layers, newLayer];
					const { params, memory, layersWithSizes } =
						calculateModelStats(newLayers);

					return {
						layers: layersWithSizes,
						totalParams: params,
						estimatedMemory: memory,
					};
				});
			},

			updateLayer: (id, updatedLayer) => {
				set((state) => {
					const newLayers = state.layers.map((layer) =>
						layer.id === id ? { ...layer, ...updatedLayer } : layer,
					);

					const { params, memory, layersWithSizes } =
						calculateModelStats(newLayers);

					return {
						layers: layersWithSizes,
						totalParams: params,
						estimatedMemory: memory,
					};
				});
			},

			removeLayer: (id) => {
				set((state) => {
					const newLayers = state.layers.filter((layer) => layer.id !== id);
					const { params, memory, layersWithSizes } =
						calculateModelStats(newLayers);

					return {
						layers: layersWithSizes,
						totalParams: params,
						estimatedMemory: memory,
					};
				});
			},

			moveLayerUp: (id) => {
				set((state) => {
					const index = state.layers.findIndex((layer) => layer.id === id);
					if (index <= 0) return state;

					const newLayers = [...state.layers];
					[newLayers[index - 1], newLayers[index]] = [
						newLayers[index],
						newLayers[index - 1],
					];

					return { layers: newLayers };
				});
			},

			moveLayerDown: (id) => {
				set((state) => {
					const index = state.layers.findIndex((layer) => layer.id === id);
					if (index === -1 || index === state.layers.length - 1) return state;

					const newLayers = [...state.layers];
					[newLayers[index], newLayers[index + 1]] = [
						newLayers[index + 1],
						newLayers[index],
					];

					return { layers: newLayers };
				});
			},

			setLearningRate: (rate) => set({ learningRate: rate }),

			setOptimizer: (optimizer) => set({ optimizer }),

			setBatchSize: (size, customSize) =>
				set({
					batchSize: size,
					customBatchSize: size === "Custom" ? customSize : undefined,
				}),

			setEpochs: (epochs) => set({ epochs }),

			setDropoutRate: (rate) => set({ dropoutRate: rate }),

			setWeightDecay: (decay) => set({ weightDecay: decay }),

			setGradientClippingThreshold: (threshold) =>
				set({ gradientClippingThreshold: threshold }),

			setLearningRateSchedule: (schedule) =>
				set({ learningRateSchedule: schedule }),

			setCustomScheduleCallback: (callback) =>
				set({ customScheduleCallback: callback }),

			toggleEarlyStopping: () =>
				set((state) => ({ useEarlyStopping: !state.useEarlyStopping })),

			toggleModelCheckpoint: () =>
				set((state) => ({ useModelCheckpoint: !state.useModelCheckpoint })),

			updateTotals: () => {
				const layers = get().layers;
				const { params, memory } = calculateModelStats(layers);
				set({ totalParams: params, estimatedMemory: memory });
			},

			setDefaultCustomFunction: (functionName) => set({ defaultCustomFunction: functionName }),
			
			addGlobalCustomFunction: (functionName: string, code: string) => {
				set((state) => {
					// Check if function already exists
					const existingIndex = state.globalCustomFunctions.findIndex(f => f.name === functionName);
					const newGlobalFunctions = [...state.globalCustomFunctions];
					
					if (existingIndex >= 0) {
						// Update existing function
						newGlobalFunctions[existingIndex] = { name: functionName, code };
					} else {
						// Add new function
						newGlobalFunctions.push({ name: functionName, code });
					}
					
					return { 
						globalCustomFunctions: newGlobalFunctions,
						defaultCustomFunction: functionName 
					};
				});
			},

			removeGlobalCustomFunction: (functionName: string) => {
				set((state) => {
					const newGlobalFunctions = state.globalCustomFunctions.filter(f => f.name !== functionName);
					
					// Also remove from layers that use this function
					const newLayers = state.layers.map(layer => {
						if (layer.selectedCustomFunction === functionName) {
							return {
								...layer,
								selectedCustomFunction: undefined,
								customActivation: undefined,
								activation: "ReLU" // Reset to default
							};
						}
						
						if (layer.customActivationFunctions) {
							return {
								...layer,
								customActivationFunctions: layer.customActivationFunctions.filter(f => f.name !== functionName)
							};
						}
						
						return layer;
					});
					
					return {
						globalCustomFunctions: newGlobalFunctions,
						layers: newLayers,
						defaultCustomFunction: state.defaultCustomFunction === functionName ? undefined : state.defaultCustomFunction
					};
				});
			},

			updateGlobalCustomFunction: (functionName: string, code: string) => {
				set((state) => {
					const newGlobalFunctions = state.globalCustomFunctions.map(f => 
						f.name === functionName ? { ...f, code } : f
					);
					
					// Also update in layers that use this function
					const newLayers = state.layers.map(layer => {
						if (layer.selectedCustomFunction === functionName) {
							return {
								...layer,
								customActivation: code
							};
						}
						
						if (layer.customActivationFunctions) {
							return {
								...layer,
								customActivationFunctions: layer.customActivationFunctions.map(f => 
									f.name === functionName ? { ...f, code } : f
								)
							};
						}
						
						return layer;
					});
					
					return {
						globalCustomFunctions: newGlobalFunctions,
						layers: newLayers
					};
				});
			},
			
			resetHyperparameters: () => set(DEFAULT_SETTINGS),
		}),
		{
			name: "hyperparameters-storage",
			storage: createJSONStorage(() => localForage),
		},
	),
);

// Initialize the store by calculating initial stats
// This will run when the module is imported
(() => {
	const store = useHyperparametersStore.getState();
	const { params, memory, layersWithSizes } = calculateModelStats(store.layers);
	useHyperparametersStore.setState({
		layers: layersWithSizes,
		totalParams: params,
		estimatedMemory: memory,
	});
})();
