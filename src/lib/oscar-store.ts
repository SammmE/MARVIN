/**
 * OscarStore - Unified Store for the Oscar Neural Network Visualization App
 * 
 * This store consolidates all previous separate stores into one unified state management solution:
 * - DataStore (data-store.ts) - Dataset and data visualization state
 * - ModelStore (model-store.ts) - Model and visualization state  
 * - TrainingStore (training-store.ts) - Training state and metrics
 * - HyperparametersStore (hyperparameters-store.ts) - Model configuration
 * 
 * Benefits of unified store:
 * - Single source of truth for all application state
 * - No need to transfer data between stores
 * - Reduced memory usage and complexity
 * - Consistent persistence strategy
 * - Better type safety with shared types
 * 
 * Migration Guide:
 * - Replace individual store imports with useOscarStore
 * - Use compatibility hooks (useDataStore, useModelStore, etc.) for gradual migration
 * - All state is available through a single store instance
 * - Persistence automatically handles all state partitions
 */

import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import localForage from "localforage";
import * as tf from "@tensorflow/tfjs";

// =============================================================================
// DATA TYPES - Combined from all stores
// =============================================================================

// Data Store Types
export type DataPoint = Record<string, unknown>;
export type Dataset = {
	data: DataPoint[];
	columns: string[];
};

export type DataStats = {
	mean: Record<string, number>;
	std: Record<string, number>;
	min: Record<string, number>;
	max: Record<string, number>;
};

export type ChartType = "scatter" | "line" | "histogram" | "pairplot";

export type PresetType =
	| "sinusoid"
	| "spiral"
	| "moons"
	| "iris"
	| "mnist-subset"
	| null;

// Model Store Types
export type ViewMode = "layer" | "neuron" | "detailed";
export type PlayState = "playing" | "paused";
export type ZoomLevel = "overview" | "layer" | "neuron";

export interface LayerVisualization {
  id: string;
  name: string;
  type: string;
  activations: number[][];
  weights?: number[][];
  biases?: number[];
  shape: number[];
  position: { x: number; y: number };
  isInput?: boolean;
  isOutput?: boolean;
  stats: {
    mean: number;
    max: number;
    min: number;
    std: number;
  };
}

export interface NeuronNode {
  id: string;
  layerId: string;
  index: number;
  activation: number;
  position: { x: number; y: number };
  type: "neuron";
}

export interface LayerNode {
  id: string;
  name: string;
  type: string;
  activationMagnitude: number;
  shape: number[];
  position: { x: number; y: number };
  isInput?: boolean;
  isOutput?: boolean;
  isExpanded?: boolean;
  stats: {
    mean: number;
    max: number;
    min: number;
    std: number;
  };
}

export interface WeightEdge {
  id: string;
  source: string;
  target: string;
  weight: number;
  thickness: number;
  animated?: boolean;
}

export interface ModelSnapshot {
  id: string;
  timestamp: number;
  epoch: number;
  loss: number;
  accuracy?: number;
  layers: LayerVisualization[];
  dataUrl?: string;
}

export interface TrainingMetrics {
  epoch: number;
  loss: number;
  accuracy?: number;
  valLoss?: number;
  valAccuracy?: number;
  timestamp: number;
}

// Training Store Types
export type TrainingState = "idle" | "training" | "paused" | "stopped" | "completed" | "error";
export type TrainingSpeed = number;

export type MetricData = {
  epoch: number;
  loss: number;
  accuracy?: number;
  valLoss?: number;
  valAccuracy?: number;
  timestamp: number;
};

export type LayerActivationData = {
  layerId: string;
  layerName: string;
  activations: number[];
  gradients?: number[];
  weights?: number[][];
  biases?: number[];
};

// Hyperparameters Store Types
export type LayerType = "Dense" | "Conv1D" | "Conv2D" | "Flatten";
export type BuiltinActivationType = "ReLU" | "Sigmoid" | "Tanh" | "Linear";
export type ActivationType = BuiltinActivationType | "Custom" | string;
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
	customActivation?: string;
	customActivationFunctions?: CustomActivationFunction[];
	selectedCustomFunction?: string;
	inputSize?: number;
	outputSize?: number;
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

// Data Store Helpers
const generateSinusoid = (): Dataset => {
	const data = Array.from({ length: 100 }, (_, i) => {
		const x = (i / 99) * 4 * Math.PI;
		const y = Math.sin(x) + Math.random() * 0.1;
		return { x, y };
	});
	return { data, columns: ["x", "y"] };
};

const generateSpiral = (): Dataset => {
	const data: DataPoint[] = [];
	for (let i = 0; i < 100; i++) {
		const r = i / 99;
		const t = 1.75 * i / 99 * 2 * Math.PI;
		const x = r * Math.cos(t) + Math.random() * 0.1;
		const y = r * Math.sin(t) + Math.random() * 0.1;
		data.push({ x, y, label: 0 });
		const x2 = r * Math.cos(t + Math.PI) + Math.random() * 0.1;
		const y2 = r * Math.sin(t + Math.PI) + Math.random() * 0.1;
		data.push({ x: x2, y: y2, label: 1 });
	}
	return { data, columns: ["x", "y", "label"] };
};

const generateMoons = (): Dataset => {
	const data: DataPoint[] = [];
	for (let i = 0; i < 100; i++) {
		const t = Math.PI * i / 99;
		const x = Math.cos(t) + Math.random() * 0.1;
		const y = Math.sin(t) + Math.random() * 0.1;
		data.push({ x, y, label: 0 });
		const x2 = 1 - Math.cos(t) + Math.random() * 0.1;
		const y2 = 1 - Math.sin(t) - 0.5 + Math.random() * 0.1;
		data.push({ x: x2, y: y2, label: 1 });
	}
	return { data, columns: ["x", "y", "label"] };
};

const generateIris = (): Dataset => {
	const data: DataPoint[] = [];
	for (let i = 0; i < 150; i++) {
		const species = Math.floor(i / 50);
		const sepalLength = 4.5 + species * 1.5 + Math.random() * 1.5;
		const sepalWidth = 2.5 + Math.random() * 1.5;
		const petalLength = 1 + species * 2 + Math.random() * 2;
		const petalWidth = 0.1 + species * 0.8 + Math.random() * 0.8;
		data.push({
			sepalLength,
			sepalWidth,
			petalLength,
			petalWidth,
			species
		});
	}
	return { data, columns: ["sepalLength", "sepalWidth", "petalLength", "petalWidth", "species"] };
};

const generateMNISTSubset = (): Dataset => {
	const data: DataPoint[] = [];
	for (let i = 0; i < 100; i++) {
		const label = Math.floor(Math.random() * 10);
		const pixels: Record<string, number> = {};
		for (let j = 0; j < 784; j++) {
			pixels[`pixel_${j}`] = Math.random();
		}
		data.push({ ...pixels, label });
	}
	const columns = Array.from({ length: 784 }, (_, i) => `pixel_${i}`);
	columns.push("label");
	return { data, columns };
};

// Model Store Helpers
const calculateActivationMagnitude = (activations: number[][]): number => {
  if (!activations || activations.length === 0) return 0;
  const flatActivations = activations.flat();
  const sum = flatActivations.reduce((acc, val) => acc + Math.abs(val), 0);
  return sum / flatActivations.length;
};

const calculateStats = (values: number[]) => {
  if (values.length === 0) return { mean: 0, max: 0, min: 0, std: 0 };
  
  const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
  const max = Math.max(...values);
  const min = Math.min(...values);
  const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
  const std = Math.sqrt(variance);
  
  return { mean, max, min, std };
};

const generateLayerNodes = (layers: LayerVisualization[]): LayerNode[] => {
  const nodes = layers.map((layer, index) => ({
    id: layer.id,
    name: layer.name,
    type: layer.type,
    activationMagnitude: calculateActivationMagnitude(layer.activations),
    shape: layer.shape,
    position: { x: index * 200, y: 100 },
    isInput: index === 0,
    isOutput: index === layers.length - 1,
    isExpanded: false,
    stats: layer.stats,
  }));
  
  return nodes;
};

const generateNeuronNodes = (layers: LayerVisualization[], expandedLayers: Set<string>): NeuronNode[] => {
  const nodes: NeuronNode[] = [];
  
  layers.forEach((layer, layerIndex) => {
    if (!expandedLayers.has(layer.id)) return;
    
    const neuronsInLayer = layer.shape && layer.shape.length > 0 
      ? layer.shape[layer.shape.length - 1]
      : 1;
    
    const maxNeuronsToShow = Math.min(neuronsInLayer, 20);
    
    if (layer.activations && layer.activations.length > 0) {
      layer.activations[0].slice(0, maxNeuronsToShow).forEach((activation, neuronIndex) => {
        nodes.push({
          id: `${layer.id}-neuron-${neuronIndex}`,
          layerId: layer.id,
          index: neuronIndex,
          activation,
          position: { x: layerIndex * 200, y: neuronIndex * 30 + 50 },
          type: "neuron",
        });
      });
    }
  });
  
  return nodes;
};

// =============================================================================
// UNIFIED OSCAR STORE
// =============================================================================

interface OscarStore {
  // =============================================================================
  // DATA STATE
  // =============================================================================
  dataset: Dataset | null;
  selectedFeatures: string[];
  selectedTarget: string | null;
  chartType: ChartType;
  xAxis: string | null;
  yAxis: string | null;
  brushEnabled: boolean;
  presetType: PresetType;
  stats: DataStats | null;

  // =============================================================================
  // MODEL STATE
  // =============================================================================
  model: tf.Sequential | null;
  isTraining: boolean;
  currentEpoch: number;
  totalEpochs: number;
  viewMode: ViewMode;
  playState: PlayState;
  focusedLayerId: string | null;
  zoomLevel: number;
  currentZoomMode: ZoomLevel;
  expandedLayers: Set<string>;
  layers: LayerVisualization[];
  layerNodes: LayerNode[];
  neuronNodes: NeuronNode[];
  edges: WeightEdge[];
  trainingMetrics: TrainingMetrics[];
  currentMetrics: TrainingMetrics | null;
  snapshots: ModelSnapshot[];
  liveUpdateEnabled: boolean;
  updateInterval: number;

  // =============================================================================
  // TRAINING STATE
  // =============================================================================
  trainingState: TrainingState;
  speed: TrainingSpeed;
  currentBatch: number;
  totalBatches: number;
  lastError: string | null;
  errorType: 'shape_mismatch' | 'general' | null;
  layerActivations: LayerActivationData[];
  modelPredictions: Array<{
    index: number;
    prediction: number;
    input: number;
    actual: number;
  }>;
  worker: Worker | null;
  trainingConfig: {
    epochs: number;
    batchSize: number;
    learningRate: number;
    optimizer: string;
    validationSplit: number;
  };
  modelConfig: {
    layers: Array<{
      type: string;
      units?: number;
      activation?: string;
      inputShape?: number[];
    }>;
    learningRate: number;
    loss: string;
    metrics: string[];
  } | null;
  trainingDataset: {
    xs: number[][];
    ys: number[][];
  } | null;

  // =============================================================================
  // HYPERPARAMETERS STATE
  // =============================================================================
  hyperparameterLayers: Layer[];
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
  defaultCustomFunction?: string;
  globalCustomFunctions: CustomActivationFunction[];

  // =============================================================================
  // DATA ACTIONS
  // =============================================================================
  setDataset: (dataset: Dataset) => void;
  setSelectedFeatures: (features: string[]) => void;
  setSelectedTarget: (target: string | null) => void;
  setChartType: (chartType: ChartType) => void;
  setXAxis: (column: string | null) => void;
  setYAxis: (column: string | null) => void;
  toggleBrush: () => void;
  setPresetType: (preset: PresetType) => void;
  loadPreset: (preset: PresetType) => void;
  calculateStats: () => void;
  clearAllData: () => void;
  clearData: () => void;

  // =============================================================================
  // MODEL ACTIONS
  // =============================================================================
  setModel: (model: tf.Sequential | null) => void;
  setTrainingState: (isTraining: boolean) => void;
  setCurrentEpoch: (epoch: number) => void;
  setTotalEpochs: (epochs: number) => void;
  setViewMode: (mode: ViewMode) => void;
  setPlayState: (state: PlayState) => void;
  setFocusedLayer: (layerId: string | null) => void;
  setZoomLevel: (zoom: number) => void;
  setZoomMode: (mode: ZoomLevel) => void;
  toggleLayerExpansion: (layerId: string) => void;
  expandLayer: (layerId: string) => void;
  collapseLayer: (layerId: string) => void;
  collapseAllLayers: () => void;
  setLayers: (layers: LayerVisualization[]) => void;
  updateLayerActivations: (layerId: string, activations: number[][]) => void;
  updateLayerWeights: (layerId: string, weights: number[][], biases?: number[]) => void;
  addTrainingMetric: (metric: TrainingMetrics) => void;
  clearTrainingMetrics: () => void;
  createSnapshot: () => Promise<string>;
  deleteSnapshot: (id: string) => void;
  clearSnapshots: () => void;
  toggleLiveUpdates: () => void;
  setUpdateInterval: (interval: number) => void;
  initializeVisualization: () => void;
  updateVisualizationData: () => void;
  resetModel: () => void;
  resetTraining: () => void;

  // =============================================================================
  // TRAINING ACTIONS
  // =============================================================================
  setTrainingStateAction: (state: TrainingState) => void;
  setSpeed: (speed: TrainingSpeed) => void;
  setTrainingConfig: (config: Partial<OscarStore['trainingConfig']>) => void;
  setModelConfig: (config: OscarStore['modelConfig']) => void;
  setTrainingDataset: (xs: number[][], ys: number[][]) => void;
  startTraining: () => Promise<void>;
  pauseTraining: () => void;
  resumeTraining: () => void;
  stopTraining: () => void;
  stepBatch: () => void;
  stepEpoch: () => void;
  scrubToEpoch: (epoch: number) => void;
  addMetric: (metric: MetricData) => void;
  updateActivations: (activations: LayerActivationData[]) => void;
  updatePredictions: (predictions: Array<{index: number; prediction: number; input: number; actual: number}>) => void;
  clearMetrics: () => void;
  setError: (error: string, type: 'shape_mismatch' | 'general') => void;
  clearError: () => void;
  initializeWorker: () => void;
  disposeWorker: () => void;

  // =============================================================================
  // HYPERPARAMETERS ACTIONS
  // =============================================================================
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

// Default hyperparameter settings
const DEFAULT_HYPERPARAMETERS = {
	hyperparameterLayers: [
		{
			id: "layer-1",
			type: "Dense" as LayerType,
			units: 128,
			activation: "ReLU" as ActivationType,
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
			type: "Dense" as LayerType,
			units: 64,
			activation: "ReLU" as ActivationType,
		},
		{
			id: "layer-3",
			type: "Dense" as LayerType,
			units: 1,
			activation: "Linear" as ActivationType,
		},
	],
	learningRate: 0.001,
	optimizer: "Adam" as OptimizerType,
	batchSize: 32 as BatchSizeOption,
	epochs: 100,
	dropoutRate: 0.2,
	weightDecay: 0.0001,
	gradientClippingThreshold: 1.0,
	learningRateSchedule: "Step" as LearningRateScheduleType,
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

export const useOscarStore = create<OscarStore>()(
  persist(
    (set, get) => ({
      // =============================================================================
      // INITIAL STATE
      // =============================================================================
      
      // Data State
      dataset: null,
      selectedFeatures: [],
      selectedTarget: null,
      chartType: "scatter",
      xAxis: null,
      yAxis: null,
      brushEnabled: false,
      presetType: null,
      stats: null,

      // Model State
      model: null,
      isTraining: false,
      currentEpoch: 0,
      totalEpochs: 100,
      viewMode: "layer",
      playState: "paused",
      focusedLayerId: null,
      zoomLevel: 1,
      currentZoomMode: "overview",
      expandedLayers: new Set<string>(),
      layers: [],
      layerNodes: [],
      neuronNodes: [],
      edges: [],
      trainingMetrics: [],
      currentMetrics: null,
      snapshots: [],
      liveUpdateEnabled: true,
      updateInterval: 500,

      // Training State
      trainingState: "idle",
      speed: 1,
      currentBatch: 0,
      totalBatches: 0,
      lastError: null,
      errorType: null,
      layerActivations: [],
      modelPredictions: [],
      worker: null,
      trainingConfig: {
        epochs: 100,
        batchSize: 32,
        learningRate: 0.001,
        optimizer: "adam",
        validationSplit: 0.2,
      },
      modelConfig: null,
      trainingDataset: null,

      // Hyperparameters State
      ...DEFAULT_HYPERPARAMETERS,

      // =============================================================================
      // DATA ACTIONS
      // =============================================================================
      
      setDataset: (dataset) => {
        set({ dataset });
        get().calculateStats();
      },

      setSelectedFeatures: (features) => set({ selectedFeatures: features }),

      setSelectedTarget: (target) => set({ selectedTarget: target }),

      setChartType: (chartType) => set({ chartType }),

      setXAxis: (column) => set({ xAxis: column }),

      setYAxis: (column) => set({ yAxis: column }),

      toggleBrush: () => set((state) => ({ brushEnabled: !state.brushEnabled })),

      setPresetType: (preset) => set({ presetType: preset }),

      loadPreset: (preset) => {
        if (preset === null) {
          // Use the dedicated clear function
          get().clearAllData();
          return;
        }

        let dataset: Dataset;
        switch (preset) {
          case "sinusoid":
            dataset = generateSinusoid();
            break;
          case "spiral":
            dataset = generateSpiral();
            break;
          case "moons":
            dataset = generateMoons();
            break;
          case "iris":
            dataset = generateIris();
            break;
          case "mnist-subset":
            dataset = generateMNISTSubset();
            break;
          default:
            return;
        }
        set({ dataset, presetType: preset });
        get().calculateStats();
        if (dataset.columns.length > 0) {
          set({
            xAxis: dataset.columns[0],
            yAxis: dataset.columns.length > 1 ? dataset.columns[1] : null,
            selectedFeatures: dataset.columns.slice(0, dataset.columns.length - 1),
            selectedTarget: dataset.columns[dataset.columns.length - 1],
          });
        }
      },

      calculateStats: () => {
        const { dataset } = get();
        if (!dataset || dataset.data.length === 0) {
          set({ stats: null });
          return;
        }

        const mean: Record<string, number> = {};
        const std: Record<string, number> = {};
        const min: Record<string, number> = {};
        const max: Record<string, number> = {};

        dataset.columns.forEach((col) => {
          if (typeof dataset.data[0][col] !== "number") return;

          const values = dataset.data
            .map((d) => d[col])
            .filter((v) => typeof v === "number");

          const sum = values.reduce((acc, val) => acc + val, 0);
          const meanVal = sum / values.length;
          mean[col] = meanVal;

          const squaredDiffs = values.map((val) => Math.pow(val - meanVal, 2));
          const avgSquaredDiff =
            squaredDiffs.reduce((acc, val) => acc + val, 0) / values.length;
          std[col] = Math.sqrt(avgSquaredDiff);

          min[col] = Math.min(...values);
          max[col] = Math.max(...values);
        });

        set({ stats: { mean, std, min, max } });
      },

      clearAllData: () => {
        // Clear all data and reset to initial state
        const { worker } = get();
        if (worker) {
          worker.terminate();
        }
        
        set({
          // Clear data state
          dataset: null,
          selectedFeatures: [],
          selectedTarget: null,
          xAxis: null,
          yAxis: null,
          presetType: null,
          stats: null,
          brushEnabled: false,
          chartType: "scatter",
          
          // Reset training state
          trainingState: "idle",
          currentEpoch: 0,
          currentBatch: 0,
          totalBatches: 0,
          lastError: null,
          errorType: null,
          layerActivations: [],
          modelPredictions: [],
          worker: null,
          trainingMetrics: [],
          trainingDataset: null,
          
          // Reset model state
          model: null,
          isTraining: false,
          layers: [],
          layerNodes: [],
          neuronNodes: [],
          edges: [],
          currentMetrics: null,
          snapshots: [],
          expandedLayers: new Set(),
        });
      },

      clearData: () => {
        // Clear only data-related state, keep model and training settings
        set({
          dataset: null,
          selectedFeatures: [],
          selectedTarget: null,
          xAxis: null,
          yAxis: null,
          presetType: null,
          stats: null,
          brushEnabled: false,
          chartType: "scatter",
        });
      },

      // =============================================================================
      // MODEL ACTIONS
      // =============================================================================

      setModel: (model) => set({ model }),

      setTrainingState: (isTraining) => set({ isTraining }),

      setCurrentEpoch: (epoch) => set({ currentEpoch: epoch }),

      setTotalEpochs: (epochs) => set({ totalEpochs: epochs }),

      setViewMode: (mode) => set({ viewMode: mode }),

      setPlayState: (state) => set({ playState: state }),

      setFocusedLayer: (layerId) => set({ focusedLayerId: layerId }),

      setZoomLevel: (zoom) => set({ zoomLevel: zoom }),

      setZoomMode: (mode) => set({ currentZoomMode: mode }),

      toggleLayerExpansion: (layerId) => {
        const { expandedLayers } = get();
        const newExpanded = new Set(expandedLayers);
        if (newExpanded.has(layerId)) {
          newExpanded.delete(layerId);
        } else {
          newExpanded.add(layerId);
        }
        set({ expandedLayers: newExpanded });
        get().updateVisualizationData();
      },

      expandLayer: (layerId) => {
        const { expandedLayers } = get();
        const newExpanded = new Set(expandedLayers);
        newExpanded.add(layerId);
        set({ expandedLayers: newExpanded });
        get().updateVisualizationData();
      },

      collapseLayer: (layerId) => {
        const { expandedLayers } = get();
        const newExpanded = new Set(expandedLayers);
        newExpanded.delete(layerId);
        set({ expandedLayers: newExpanded });
        get().updateVisualizationData();
      },

      collapseAllLayers: () => {
        set({ expandedLayers: new Set() });
        get().updateVisualizationData();
      },

      setLayers: (layers) => {
        set({ layers });
        get().updateVisualizationData();
      },

      updateLayerActivations: (layerId, activations) => {
        const { layers } = get();
        const updatedLayers = layers.map(layer => 
          layer.id === layerId 
            ? { ...layer, activations, stats: calculateStats(activations.flat()) }
            : layer
        );
        set({ layers: updatedLayers });
        get().updateVisualizationData();
      },

      updateLayerWeights: (layerId, weights, biases) => {
        const { layers } = get();
        const updatedLayers = layers.map(layer => 
          layer.id === layerId 
            ? { ...layer, weights, biases }
            : layer
        );
        set({ layers: updatedLayers });
      },

      addTrainingMetric: (metric) => {
        set((state) => ({
          trainingMetrics: [...state.trainingMetrics, metric],
          currentMetrics: metric,
        }));
      },

      clearTrainingMetrics: () => set({ trainingMetrics: [], currentMetrics: null }),

      createSnapshot: async () => {
        const { layers, currentEpoch, currentMetrics } = get();
        const id = `snapshot-${Date.now()}`;
        const snapshot: ModelSnapshot = {
          id,
          timestamp: Date.now(),
          epoch: currentEpoch,
          loss: currentMetrics?.loss || 0,
          accuracy: currentMetrics?.accuracy,
          layers: JSON.parse(JSON.stringify(layers)),
        };
        
        set((state) => ({
          snapshots: [...state.snapshots, snapshot],
        }));
        
        return id;
      },

      deleteSnapshot: (id) => {
        set((state) => ({
          snapshots: state.snapshots.filter(s => s.id !== id),
        }));
      },

      clearSnapshots: () => set({ snapshots: [] }),

      toggleLiveUpdates: () => {
        set((state) => ({ liveUpdateEnabled: !state.liveUpdateEnabled }));
      },

      setUpdateInterval: (interval) => set({ updateInterval: interval }),

      initializeVisualization: () => {
        get().updateVisualizationData();
      },

      updateVisualizationData: () => {
        const { layers, expandedLayers } = get();
        const layerNodes = generateLayerNodes(layers);
        const neuronNodes = generateNeuronNodes(layers, expandedLayers);
        set({ layerNodes, neuronNodes });
      },

      resetModel: () => {
        set({
          model: null,
          isTraining: false,
          currentEpoch: 0,
          layers: [],
          layerNodes: [],
          neuronNodes: [],
          edges: [],
          trainingMetrics: [],
          currentMetrics: null,
          snapshots: [],
          expandedLayers: new Set(),
          trainingState: "idle",
        });
      },

      resetTraining: () => {
        set({
          trainingState: "idle",
          currentEpoch: 0,
          currentBatch: 0,
          totalBatches: 0,
          trainingMetrics: [],
          currentMetrics: null,
        });
      },

      // =============================================================================
      // TRAINING ACTIONS
      // =============================================================================

      setTrainingStateAction: (state) => set({ trainingState: state }),

      setSpeed: (speed) => set({ speed }),

      setTrainingConfig: (config) => {
        set((state) => ({
          trainingConfig: { ...state.trainingConfig, ...config },
        }));
      },

      setModelConfig: (config) => set({ modelConfig: config }),

      setTrainingDataset: (xs, ys) => set({ trainingDataset: { xs, ys } }),

      startTraining: async () => {
        const state = get();
        const { worker, modelConfig, trainingDataset, trainingConfig } = state;
        
        if (!worker) {
          console.error("No worker available for training");
          set({ lastError: "Training worker not initialized", errorType: "general" });
          return;
        }
        
        if (!trainingDataset?.xs || !trainingDataset?.ys) {
          console.error("No training data available");
          set({ lastError: "No training data available", errorType: "general" });
          return;
        }
        
        if (!modelConfig) {
          console.error("No model configuration available");
          set({ lastError: "No model configuration available", errorType: "general" });
          return;
        }
        
        console.log("Starting training with:", {
          dataSize: trainingDataset.xs.length,
          modelLayers: modelConfig.layers?.length || 0,
          epochs: trainingConfig.epochs,
          batchSize: trainingConfig.batchSize
        });
        
        // Send training configuration to worker
        const trainingMessage = {
          type: 'start',
          payload: {
            modelConfig,
            dataConfig: {
              xs: trainingDataset.xs,
              ys: trainingDataset.ys,
              validationSplit: 0.2
            },
            trainingConfig,
            speed: state.speed
          }
        };
        
        worker.postMessage(trainingMessage);
        set({ trainingState: "training" });
      },

      pauseTraining: () => {
        const { worker } = get();
        if (worker) {
          worker.postMessage({ type: 'pause' });
        }
        set({ trainingState: "paused" });
      },

      resumeTraining: () => {
        const { worker } = get();
        if (worker) {
          worker.postMessage({ type: 'resume' });
        }
        set({ trainingState: "training" });
      },

      stopTraining: () => {
        const { worker } = get();
        if (worker) {
          worker.postMessage({ type: 'stop' });
          worker.terminate();
        }
        set({ trainingState: "idle", worker: null });
      },

      stepBatch: () => {
        const { worker } = get();
        if (worker) {
          worker.postMessage({ type: 'step', payload: { stepType: 'batch' } });
        }
      },

      stepEpoch: () => {
        const { worker } = get();
        if (worker) {
          worker.postMessage({ type: 'step', payload: { stepType: 'epoch' } });
        }
      },

      scrubToEpoch: (epoch) => {
        set({ currentEpoch: epoch });
      },

      addMetric: (metric) => {
        set((state) => ({
          trainingMetrics: [...state.trainingMetrics, metric],
        }));
      },

      updateActivations: (activations) => set({ layerActivations: activations }),

      updatePredictions: (predictions) => set({ modelPredictions: predictions }),

      clearMetrics: () => set({ trainingMetrics: [] }),

      setError: (error, type) => set({ lastError: error, errorType: type }),

      clearError: () => set({ lastError: null, errorType: null }),

      initializeWorker: () => {
        const worker = new Worker(new URL('../workers/training-worker.ts', import.meta.url), {
          type: 'module'
        });
        
        // Set up message handler for worker responses
        worker.onmessage = (event) => {
          const message = event.data;
          console.log("Worker message received:", message);
          
          switch (message.type) {
            case 'progress':
              set((state) => ({
                currentEpoch: message.payload.epoch || state.currentEpoch,
                currentBatch: message.payload.batch || state.currentBatch,
                totalBatches: message.payload.totalBatches || state.totalBatches
              }));
              break;
              
            case 'metrics':
              get().addMetric(message.payload);
              break;
              
            case 'activations':
              get().updateActivations(message.payload);
              break;
              
            case 'predictions':
              get().updatePredictions(message.payload);
              break;
              
            case 'complete':
              set({ trainingState: "completed" });
              console.log("Training completed");
              break;
              
            case 'error':
              console.error("Training error:", message.payload);
              set({ 
                trainingState: "idle",
                lastError: message.payload.message || "Training error occurred",
                errorType: "general"
              });
              break;
              
            case 'paused':
              set({ trainingState: "paused" });
              break;
              
            default:
              console.log("Unknown worker message type:", message.type);
          }
        };
        
        worker.onerror = (error) => {
          console.error("Worker error:", error);
          set({ 
            lastError: "Training worker error: " + error.message,
            errorType: "general",
            trainingState: "idle"
          });
        };
        
        set({ worker });
      },

      disposeWorker: () => {
        const { worker } = get();
        if (worker) {
          worker.terminate();
          set({ worker: null });
        }
      },

      // =============================================================================
      // HYPERPARAMETERS ACTIONS
      // =============================================================================

      addLayer: (layer) => {
        const newLayer = { ...layer, id: `layer-${Date.now()}` };
        set((state) => ({
          hyperparameterLayers: [...state.hyperparameterLayers, newLayer],
        }));
        get().updateTotals();
      },

      updateLayer: (id, layer) => {
        set((state) => ({
          hyperparameterLayers: state.hyperparameterLayers.map((l) =>
            l.id === id ? { ...l, ...layer } : l
          ),
        }));
        get().updateTotals();
      },

      removeLayer: (id) => {
        set((state) => ({
          hyperparameterLayers: state.hyperparameterLayers.filter((l) => l.id !== id),
        }));
        get().updateTotals();
      },

      moveLayerUp: (id) => {
        set((state) => {
          const index = state.hyperparameterLayers.findIndex((l) => l.id === id);
          if (index > 0) {
            const newLayers = [...state.hyperparameterLayers];
            [newLayers[index - 1], newLayers[index]] = [newLayers[index], newLayers[index - 1]];
            return { hyperparameterLayers: newLayers };
          }
          return state;
        });
      },

      moveLayerDown: (id) => {
        set((state) => {
          const index = state.hyperparameterLayers.findIndex((l) => l.id === id);
          if (index < state.hyperparameterLayers.length - 1) {
            const newLayers = [...state.hyperparameterLayers];
            [newLayers[index], newLayers[index + 1]] = [newLayers[index + 1], newLayers[index]];
            return { hyperparameterLayers: newLayers };
          }
          return state;
        });
      },

      setLearningRate: (rate) => set({ learningRate: rate }),

      setOptimizer: (optimizer) => set({ optimizer }),

      setBatchSize: (size, customSize) => {
        set({ batchSize: size });
        if (customSize !== undefined) {
          set({ customBatchSize: customSize });
        }
      },

      setEpochs: (epochs) => set({ epochs }),

      setDropoutRate: (rate) => set({ dropoutRate: rate }),

      setDefaultCustomFunction: (name) => set({ defaultCustomFunction: name }),

      addGlobalCustomFunction: (functionName, code) => {
        set((state) => ({
          globalCustomFunctions: [
            ...state.globalCustomFunctions,
            { name: functionName, code },
          ],
        }));
      },

      removeGlobalCustomFunction: (functionName) => {
        set((state) => ({
          globalCustomFunctions: state.globalCustomFunctions.filter(
            (f) => f.name !== functionName
          ),
        }));
      },

      updateGlobalCustomFunction: (functionName, code) => {
        set((state) => ({
          globalCustomFunctions: state.globalCustomFunctions.map((f) =>
            f.name === functionName ? { ...f, code } : f
          ),
        }));
      },

      setWeightDecay: (decay) => set({ weightDecay: decay }),

      setGradientClippingThreshold: (threshold) => set({ gradientClippingThreshold: threshold }),

      setLearningRateSchedule: (schedule) => set({ learningRateSchedule: schedule }),

      setCustomScheduleCallback: (callback) => set({ customScheduleCallback: callback }),

      toggleEarlyStopping: () => {
        set((state) => ({ useEarlyStopping: !state.useEarlyStopping }));
      },

      toggleModelCheckpoint: () => {
        set((state) => ({ useModelCheckpoint: !state.useModelCheckpoint }));
      },

      updateTotals: () => {
        const { hyperparameterLayers } = get();
        let totalParams = 0;
        let previousLayerSize = 0;

        hyperparameterLayers.forEach((layer, index) => {
          if (layer.type === "Dense") {
            const currentLayerSize = layer.units || 0;
            if (index > 0) {
              totalParams += previousLayerSize * currentLayerSize + currentLayerSize;
            }
            previousLayerSize = currentLayerSize;
          }
        });

        const estimatedMemory = totalParams * 4; // 4 bytes per parameter (float32)
        set({ totalParams, estimatedMemory });
      },

      resetHyperparameters: () => {
        set(DEFAULT_HYPERPARAMETERS);
      },
    }),
    {
      name: "oscar-store",
      storage: createJSONStorage(() => localForage),
      partialize: (state) => ({
        // Persist only non-transient state
        dataset: state.dataset,
        selectedFeatures: state.selectedFeatures,
        selectedTarget: state.selectedTarget,
        chartType: state.chartType,
        xAxis: state.xAxis,
        yAxis: state.yAxis,
        presetType: state.presetType,
        stats: state.stats,
        viewMode: state.viewMode,
        zoomLevel: state.zoomLevel,
        currentZoomMode: state.currentZoomMode,
        liveUpdateEnabled: state.liveUpdateEnabled,
        updateInterval: state.updateInterval,
        trainingConfig: state.trainingConfig,
        hyperparameterLayers: state.hyperparameterLayers,
        learningRate: state.learningRate,
        optimizer: state.optimizer,
        batchSize: state.batchSize,
        customBatchSize: state.customBatchSize,
        epochs: state.epochs,
        dropoutRate: state.dropoutRate,
        weightDecay: state.weightDecay,
        gradientClippingThreshold: state.gradientClippingThreshold,
        learningRateSchedule: state.learningRateSchedule,
        useEarlyStopping: state.useEarlyStopping,
        useModelCheckpoint: state.useModelCheckpoint,
        customScheduleCallback: state.customScheduleCallback,
        defaultCustomFunction: state.defaultCustomFunction,
        globalCustomFunctions: state.globalCustomFunctions,
      }),
    }
  )
);

// =============================================================================
// LEGACY COMPATIBILITY EXPORTS
// =============================================================================

// Export individual store hooks for backward compatibility during migration
export const useDataStore = () => {
  const store = useOscarStore();
  return {
    dataset: store.dataset,
    selectedFeatures: store.selectedFeatures,
    selectedTarget: store.selectedTarget,
    chartType: store.chartType,
    xAxis: store.xAxis,
    yAxis: store.yAxis,
    brushEnabled: store.brushEnabled,
    presetType: store.presetType,
    stats: store.stats,
    setDataset: store.setDataset,
    setSelectedFeatures: store.setSelectedFeatures,
    setSelectedTarget: store.setSelectedTarget,
    setChartType: store.setChartType,
    setXAxis: store.setXAxis,
    setYAxis: store.setYAxis,
    toggleBrush: store.toggleBrush,
    setPresetType: store.setPresetType,
    loadPreset: store.loadPreset,
    calculateStats: store.calculateStats,
    clearAllData: store.clearAllData,
    clearData: store.clearData,
  };
};

export const useModelStore = () => {
  const store = useOscarStore();
  return {
    model: store.model,
    isTraining: store.isTraining,
    currentEpoch: store.currentEpoch,
    totalEpochs: store.totalEpochs,
    viewMode: store.viewMode,
    playState: store.playState,
    focusedLayerId: store.focusedLayerId,
    zoomLevel: store.zoomLevel,
    currentZoomMode: store.currentZoomMode,
    expandedLayers: store.expandedLayers,
    layers: store.layers,
    layerNodes: store.layerNodes,
    neuronNodes: store.neuronNodes,
    edges: store.edges,
    trainingMetrics: store.trainingMetrics,
    currentMetrics: store.currentMetrics,
    snapshots: store.snapshots,
    liveUpdateEnabled: store.liveUpdateEnabled,
    updateInterval: store.updateInterval,
    setModel: store.setModel,
    setTrainingState: store.setTrainingState,
    setCurrentEpoch: store.setCurrentEpoch,
    setTotalEpochs: store.setTotalEpochs,
    setViewMode: store.setViewMode,
    setPlayState: store.setPlayState,
    setFocusedLayer: store.setFocusedLayer,
    setZoomLevel: store.setZoomLevel,
    setZoomMode: store.setZoomMode,
    toggleLayerExpansion: store.toggleLayerExpansion,
    expandLayer: store.expandLayer,
    collapseLayer: store.collapseLayer,
    collapseAllLayers: store.collapseAllLayers,
    setLayers: store.setLayers,
    updateLayerActivations: store.updateLayerActivations,
    updateLayerWeights: store.updateLayerWeights,
    addTrainingMetric: store.addTrainingMetric,
    clearTrainingMetrics: store.clearTrainingMetrics,
    createSnapshot: store.createSnapshot,
    deleteSnapshot: store.deleteSnapshot,
    clearSnapshots: store.clearSnapshots,
    toggleLiveUpdates: store.toggleLiveUpdates,
    setUpdateInterval: store.setUpdateInterval,
    initializeVisualization: store.initializeVisualization,
    updateVisualizationData: store.updateVisualizationData,
    resetModel: store.resetModel,
    resetTraining: store.resetTraining,
  };
};

export const useTrainingStore = () => {
  const store = useOscarStore();
  return {
    trainingState: store.trainingState,
    speed: store.speed,
    currentEpoch: store.currentEpoch,
    totalEpochs: store.totalEpochs,
    currentBatch: store.currentBatch,
    totalBatches: store.totalBatches,
    lastError: store.lastError,
    errorType: store.errorType,
    trainingMetrics: store.trainingMetrics,
    layerActivations: store.layerActivations,
    modelPredictions: store.modelPredictions,
    worker: store.worker,
    trainingConfig: store.trainingConfig,
    modelConfig: store.modelConfig,
    dataset: store.trainingDataset,
    setTrainingState: store.setTrainingStateAction,
    setSpeed: store.setSpeed,
    setTrainingConfig: store.setTrainingConfig,
    setModelConfig: store.setModelConfig,
    setDataset: store.setTrainingDataset,
    startTraining: store.startTraining,
    pauseTraining: store.pauseTraining,
    resumeTraining: store.resumeTraining,
    stopTraining: store.stopTraining,
    resetTraining: store.resetTraining,
    stepBatch: store.stepBatch,
    stepEpoch: store.stepEpoch,
    scrubToEpoch: store.scrubToEpoch,
    addMetric: store.addMetric,
    updateActivations: store.updateActivations,
    updatePredictions: store.updatePredictions,
    clearMetrics: store.clearMetrics,
    setError: store.setError,
    clearError: store.clearError,
    initializeWorker: store.initializeWorker,
    disposeWorker: store.disposeWorker,
  };
};

export const useHyperparametersStore = () => {
  const store = useOscarStore();
  return {
    layers: store.hyperparameterLayers,
    learningRate: store.learningRate,
    optimizer: store.optimizer,
    batchSize: store.batchSize,
    customBatchSize: store.customBatchSize,
    epochs: store.epochs,
    dropoutRate: store.dropoutRate,
    weightDecay: store.weightDecay,
    gradientClippingThreshold: store.gradientClippingThreshold,
    learningRateSchedule: store.learningRateSchedule,
    useEarlyStopping: store.useEarlyStopping,
    useModelCheckpoint: store.useModelCheckpoint,
    customScheduleCallback: store.customScheduleCallback,
    totalParams: store.totalParams,
    estimatedMemory: store.estimatedMemory,
    defaultCustomFunction: store.defaultCustomFunction,
    globalCustomFunctions: store.globalCustomFunctions,
    addLayer: store.addLayer,
    updateLayer: store.updateLayer,
    removeLayer: store.removeLayer,
    moveLayerUp: store.moveLayerUp,
    moveLayerDown: store.moveLayerDown,
    setLearningRate: store.setLearningRate,
    setOptimizer: store.setOptimizer,
    setBatchSize: store.setBatchSize,
    setEpochs: store.setEpochs,
    setDropoutRate: store.setDropoutRate,
    setDefaultCustomFunction: store.setDefaultCustomFunction,
    addGlobalCustomFunction: store.addGlobalCustomFunction,
    removeGlobalCustomFunction: store.removeGlobalCustomFunction,
    updateGlobalCustomFunction: store.updateGlobalCustomFunction,
    setWeightDecay: store.setWeightDecay,
    setGradientClippingThreshold: store.setGradientClippingThreshold,
    setLearningRateSchedule: store.setLearningRateSchedule,
    setCustomScheduleCallback: store.setCustomScheduleCallback,
    toggleEarlyStopping: store.toggleEarlyStopping,
    toggleModelCheckpoint: store.toggleModelCheckpoint,
    updateTotals: store.updateTotals,
    resetHyperparameters: store.resetHyperparameters,
  };
};
