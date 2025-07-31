import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import localForage from "localforage";
import * as tf from "@tensorflow/tfjs";

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

// Helper functions
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
  console.log("generateLayerNodes called with layers:", layers.map(l => ({ id: l.id, name: l.name })));
  
  const nodes = layers.map((layer, index) => ({
    id: layer.id,
    name: layer.name,
    type: layer.type,
    activationMagnitude: calculateActivationMagnitude(layer.activations),
    shape: layer.shape,
    position: { x: index * 200, y: 100 },
    isInput: index === 0, // First layer is input
    isOutput: index === layers.length - 1, // Last layer is output
    isExpanded: false,
    stats: layer.stats,
  }));
  
  console.log("Generated layer nodes:", nodes.map(n => ({ id: n.id, name: n.name, isInput: n.isInput, isOutput: n.isOutput })));
  return nodes;
};

interface ModelState {
  // Core model state
  model: tf.Sequential | null;
  isTraining: boolean;
  currentEpoch: number;
  totalEpochs: number;
  
  // Visualization state
  viewMode: ViewMode;
  playState: PlayState;
  focusedLayerId: string | null;
  zoomLevel: number;
  currentZoomMode: ZoomLevel;
  expandedLayers: Set<string>;
  
  // Data for visualization
  layers: LayerVisualization[];
  layerNodes: LayerNode[];
  neuronNodes: NeuronNode[];
  edges: WeightEdge[];
  
  // Training metrics
  trainingMetrics: TrainingMetrics[];
  currentMetrics: TrainingMetrics | null;
  
  // Snapshots
  snapshots: ModelSnapshot[];
  
  // Live updates
  liveUpdateEnabled: boolean;
  updateInterval: number;
  
  // Actions
  setModel: (model: tf.Sequential) => void;
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
}

const generateNeuronNodes = (layers: LayerVisualization[], expandedLayers: Set<string>): NeuronNode[] => {
  const nodes: NeuronNode[] = [];
  
  layers.forEach((layer, layerIndex) => {
    // Only generate neuron nodes for expanded layers
    if (!expandedLayers.has(layer.id)) return;
    
    const neuronsInLayer = layer.shape && layer.shape.length > 0 
      ? layer.shape[layer.shape.length - 1] // Get the last dimension (number of units)
      : 1;
    
    const maxNeuronsToShow = Math.min(neuronsInLayer, 20); // Limit to 20 neurons for performance
    
    if (layer.activations && layer.activations.length > 0) {
      // Use actual activations if available
      layer.activations[0].slice(0, maxNeuronsToShow).forEach((activation, neuronIndex) => {
        nodes.push({
          id: `${layer.id}-neuron-${neuronIndex}`,
          layerId: layer.id,
          index: neuronIndex,
          activation,
          position: {
            x: layerIndex * 300 + 150,
            y: 200 + neuronIndex * 40,
          },
          type: "neuron",
        });
      });
    } else {
      // Create default neurons based on layer shape
      for (let neuronIndex = 0; neuronIndex < maxNeuronsToShow; neuronIndex++) {
        nodes.push({
          id: `${layer.id}-neuron-${neuronIndex}`,
          layerId: layer.id,
          index: neuronIndex,
          activation: Math.random() * 2 - 1, // Random activation for demo
          position: {
            x: layerIndex * 300 + 150,
            y: 200 + neuronIndex * 40,
          },
          type: "neuron",
        });
      }
    }
  });
  
  return nodes;
};

const generateEdges = (layers: LayerVisualization[], viewMode: ViewMode): WeightEdge[] => {
  const edges: WeightEdge[] = [];
  
  console.log("generateEdges called:", {
    layersCount: layers.length,
    viewMode,
    layerIds: layers.map(l => l.id)
  });
  
  for (let i = 0; i < layers.length - 1; i++) {
    const currentLayer = layers[i];
    const nextLayer = layers[i + 1];
    
    console.log(`Creating edge between ${currentLayer.id} and ${nextLayer.id}`);
    
    if (viewMode === "layer") {
      // Simple layer-to-layer connection
      const meanWeight = currentLayer.weights ? Math.abs(currentLayer.stats.mean) : 0.5;
      const edge = {
        id: `edge-${currentLayer.id}-${nextLayer.id}`,
        source: currentLayer.id,
        target: nextLayer.id,
        weight: meanWeight,
        thickness: Math.min(10, Math.max(2, meanWeight * 5 + 1)),
        animated: true,
      };
      
      console.log("Created layer edge:", edge);
      edges.push(edge);
    } else if (viewMode === "neuron" && currentLayer.weights) {
      // Individual neuron connections
      currentLayer.weights.forEach((weightRow, sourceIndex) => {
        weightRow.forEach((weight, targetIndex) => {
          if (Math.abs(weight) > 0.01) { // Only show significant weights
            edges.push({
              id: `edge-${currentLayer.id}-${sourceIndex}-${nextLayer.id}-${targetIndex}`,
              source: `${currentLayer.id}-neuron-${sourceIndex}`,
              target: `${nextLayer.id}-neuron-${targetIndex}`,
              weight,
              thickness: Math.min(5, Math.max(0.5, Math.abs(weight) * 3)),
            });
          }
        });
      });
    }
  }
  
  console.log("Generated edges:", edges);
  return edges;
};

export const useModelStore = create<ModelState>()(
  persist(
    (set, get) => ({
      // Initial state
      model: null,
      isTraining: false,
      currentEpoch: 0,
      totalEpochs: 0,
      viewMode: "layer",
      playState: "paused",
      focusedLayerId: null,
      zoomLevel: 1,
      currentZoomMode: "overview",
      expandedLayers: new Set(),
      layers: [],
      layerNodes: [],
      neuronNodes: [],
      edges: [],
      trainingMetrics: [],
      currentMetrics: null,
      snapshots: [],
      liveUpdateEnabled: true,
      updateInterval: 100,

      // Actions
      setModel: (model) => set({ model }),
      
      setTrainingState: (isTraining) => set({ isTraining }),
      
      setCurrentEpoch: (currentEpoch) => set({ currentEpoch }),
      
      setTotalEpochs: (totalEpochs) => set({ totalEpochs }),
      
      setViewMode: (viewMode) => {
        set({ viewMode });
        get().updateVisualizationData();
      },
      
      setPlayState: (playState) => set({ playState }),
      
      setFocusedLayer: (focusedLayerId) => set({ focusedLayerId }),
      
      setZoomLevel: (zoomLevel) => set({ zoomLevel }),
      
      setZoomMode: (currentZoomMode) => set({ currentZoomMode }),
      
      toggleLayerExpansion: (layerId) => {
        const { expandedLayers } = get();
        const newExpandedLayers = new Set(expandedLayers);
        if (newExpandedLayers.has(layerId)) {
          newExpandedLayers.delete(layerId);
        } else {
          newExpandedLayers.add(layerId);
        }
        set({ expandedLayers: newExpandedLayers });
        get().updateVisualizationData();
      },
      
      expandLayer: (layerId) => {
        const { expandedLayers } = get();
        const newExpandedLayers = new Set(expandedLayers);
        newExpandedLayers.add(layerId);
        set({ expandedLayers: newExpandedLayers });
        get().updateVisualizationData();
      },
      
      collapseLayer: (layerId) => {
        const { expandedLayers } = get();
        const newExpandedLayers = new Set(expandedLayers);
        newExpandedLayers.delete(layerId);
        set({ expandedLayers: newExpandedLayers });
        get().updateVisualizationData();
      },
      
      collapseAllLayers: () => {
        set({ expandedLayers: new Set() });
        get().updateVisualizationData();
      },
      
      updateLayerActivations: (layerId, activations) => {
        const { layers } = get();
        const updatedLayers = layers.map(layer => {
          if (layer.id === layerId) {
            const flatActivations = activations.flat();
            const stats = calculateStats(flatActivations);
            return { ...layer, activations, stats };
          }
          return layer;
        });
        set({ layers: updatedLayers });
        get().updateVisualizationData();
      },
      
      updateLayerWeights: (layerId, weights, biases) => {
        const { layers } = get();
        const updatedLayers = layers.map(layer => {
          if (layer.id === layerId) {
            const flatWeights = weights.flat();
            const stats = calculateStats(flatWeights);
            return { ...layer, weights, biases, stats };
          }
          return layer;
        });
        set({ layers: updatedLayers });
        get().updateVisualizationData();
      },
      
      addTrainingMetric: (metric) => {
        const { trainingMetrics } = get();
        set({
          trainingMetrics: [...trainingMetrics, metric],
          currentMetrics: metric,
        });
      },
      
      clearTrainingMetrics: () => set({ trainingMetrics: [], currentMetrics: null }),
      
      createSnapshot: async () => {
        const { layers, currentEpoch, currentMetrics, snapshots } = get();
        const snapshot: ModelSnapshot = {
          id: `snapshot-${Date.now()}`,
          timestamp: Date.now(),
          epoch: currentEpoch,
          loss: currentMetrics?.loss || 0,
          accuracy: currentMetrics?.accuracy,
          layers: JSON.parse(JSON.stringify(layers)),
        };
        
        set({ snapshots: [...snapshots, snapshot] });
        return snapshot.id;
      },
      
      deleteSnapshot: (id) => {
        const { snapshots } = get();
        set({ snapshots: snapshots.filter(s => s.id !== id) });
      },
      
      clearSnapshots: () => set({ snapshots: [] }),
      
      toggleLiveUpdates: () => {
        const { liveUpdateEnabled } = get();
        set({ liveUpdateEnabled: !liveUpdateEnabled });
      },
      
      setUpdateInterval: (updateInterval) => set({ updateInterval }),
      
      initializeVisualization: () => {
        const { model } = get();
        console.log("ModelStore.initializeVisualization called with model:", !!model);
        
        if (!model) {
          console.log("No model available for visualization");
          return;
        }
        
        console.log("Model details:", {
          layerCount: model.layers.length,
          layerNames: model.layers.map(l => l.name),
          layerTypes: model.layers.map(l => l.getClassName())
        });
        
        const layers: LayerVisualization[] = model.layers.map((layer, index) => {
          const outputShape = layer.outputShape as number[];
          const isInput = index === 0;
          const isOutput = index === model.layers.length - 1;
          
          const layerViz = {
            id: `layer-${index}`,
            name: layer.name || `${isInput ? 'Input' : isOutput ? 'Output' : 'Hidden'} Layer ${index + 1}`,
            type: layer.getClassName(),
            activations: [], // Will be populated later
            weights: undefined,
            biases: undefined,
            shape: outputShape || [1],
            position: { x: index * 250 + 100, y: 100 },
            isInput,
            isOutput,
            stats: { mean: 0, max: 0, min: 0, std: 0 },
          };
          
          console.log(`Created layer visualization ${index}:`, layerViz);
          return layerViz;
        });
        
        console.log("Setting layers in store:", layers.length);
        set({ layers });
        get().updateVisualizationData();
      },
      
      updateVisualizationData: () => {
        const { layers, viewMode, expandedLayers } = get();
        console.log("updateVisualizationData called:", {
          layersCount: layers.length,
          viewMode,
          expandedLayersCount: expandedLayers.size,
          layers: layers.map(l => ({ id: l.id, name: l.name }))
        });
        
        const layerNodes = generateLayerNodes(layers);
        const neuronNodes = generateNeuronNodes(layers, expandedLayers);
        const edges = generateEdges(layers, viewMode);
        
        // Update expanded state for layer nodes
        layerNodes.forEach(node => {
          node.isExpanded = expandedLayers.has(node.id);
        });
        
        console.log("Generated visualization data:", {
          layerNodesCount: layerNodes.length,
          neuronNodesCount: neuronNodes.length,
          edgesCount: edges.length,
          layerNodes: layerNodes.map(n => ({ id: n.id, name: n.name, isExpanded: n.isExpanded })),
          edges: edges.slice(0, 3)
        });
        
        set({ layerNodes, neuronNodes, edges });
      },
      
      resetModel: () => {
        set({
          model: null,
          isTraining: false,
          currentEpoch: 0,
          totalEpochs: 0,
          layers: [],
          layerNodes: [],
          neuronNodes: [],
          edges: [],
          trainingMetrics: [],
          currentMetrics: null,
          focusedLayerId: null,
        });
      },
    }),
    {
      name: "model-store",
      storage: createJSONStorage(() => localForage),
      partialize: (state) => ({
        viewMode: state.viewMode,
        zoomLevel: state.zoomLevel,
        liveUpdateEnabled: state.liveUpdateEnabled,
        updateInterval: state.updateInterval,
        snapshots: state.snapshots,
      }),
    }
  )
);
