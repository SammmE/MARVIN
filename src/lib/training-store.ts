import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import localForage from "localforage";
import type { TrainingState, TrainingSpeed } from "../components/training-controls";
import type { MetricData } from "../components/live-metrics-chart";
import type { LayerActivationData } from "../components/activation-mini-panels";

interface TrainingStore {
    // Training state
    trainingState: TrainingState;
    speed: TrainingSpeed;
    currentEpoch: number;
    totalEpochs: number;
    currentBatch: number;
    totalBatches: number;
    
    // Training data
    trainingMetrics: MetricData[];
    layerActivations: LayerActivationData[];
    modelPredictions: Array<{
        index: number;
        prediction: number;
        input: number;
        actual: number;
    }>;
    
    // Worker management
    worker: Worker | null;
    
    // Training configuration
    trainingConfig: {
        epochs: number;
        batchSize: number;
        learningRate: number;
        optimizer: string;
        validationSplit: number;
    };
    
    // Dataset for training
    dataset: {
        xs: number[][];
        ys: number[][];
    } | null;
    
    // Actions
    setTrainingState: (state: TrainingState) => void;
    setSpeed: (speed: TrainingSpeed) => void;
    setTrainingConfig: (config: Partial<TrainingStore['trainingConfig']>) => void;
    setDataset: (xs: number[][], ys: number[][]) => void;
    
    // Training controls
    startTraining: () => Promise<void>;
    pauseTraining: () => void;
    resumeTraining: () => void;
    stopTraining: () => void;
    stepBatch: () => void;
    stepEpoch: () => void;
    scrubToEpoch: (epoch: number) => void;
    
    // Data management
    addMetric: (metric: MetricData) => void;
    updateActivations: (activations: LayerActivationData[]) => void;
    updatePredictions: (predictions: Array<{index: number; prediction: number; input: number; actual: number}>) => void;
    clearMetrics: () => void;
    
    // Worker management
    initializeWorker: () => void;
    disposeWorker: () => void;
}

export const useTrainingStore = create<TrainingStore>()(
    persist(
        (set, get) => ({
            // Initial state
            trainingState: "idle",
            speed: 1,
            currentEpoch: 0,
            totalEpochs: 100,
            currentBatch: 0,
            totalBatches: 0,
            trainingMetrics: [],
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
            dataset: null,

            // Actions
            setTrainingState: (trainingState) => set({ trainingState }),
            setSpeed: (speed) => set({ speed }),
            setTrainingConfig: (config) => set(state => ({
                trainingConfig: { ...state.trainingConfig, ...config }
            })),
            setDataset: (xs, ys) => set({ dataset: { xs, ys } }),

            // Training controls
            startTraining: async () => {
                const { dataset, trainingConfig, speed, trainingState, currentEpoch } = get();
                
                console.log("startTraining called", { 
                    hasDataset: !!dataset,
                    datasetSize: dataset ? dataset.xs.length : 0,
                    config: trainingConfig 
                });
                
                if (!dataset) {
                    console.error("Dataset not available");
                    return;
                }

                // Clear metrics before starting new training
                get().clearMetrics();

                // Dispose existing worker completely and create a new one
                get().disposeWorker();
                
                // Wait a bit for cleanup
                await new Promise(resolve => setTimeout(resolve, 200));
                
                // Initialize a fresh worker
                get().initializeWorker();
                
                // Wait for worker to be ready
                await new Promise(resolve => setTimeout(resolve, 100));
                
                const { worker } = get();
                if (!worker) {
                    console.error("Failed to create worker");
                    return;
                }

                set({ 
                    trainingState: "training",
                    // Always start from 0 for fresh training sessions
                    currentEpoch: 0,
                    currentBatch: 0,
                    totalEpochs: trainingConfig.epochs
                });

                console.log("Configuring worker...");
                // Configure worker
                worker.postMessage({
                    type: 'config',
                    payload: {
                        modelConfig: {
                            layers: [
                                { type: 'dense', units: 64, activation: 'relu', inputShape: [dataset.xs[0].length] },
                                { type: 'dense', units: 32, activation: 'relu' },
                                { type: 'dense', units: dataset.ys[0].length, activation: 'linear' }
                            ],
                            learningRate: trainingConfig.learningRate,
                            loss: 'meanSquaredError',
                            metrics: ['mae']
                        },
                        dataConfig: {
                            xs: dataset.xs,
                            ys: dataset.ys,
                            validationSplit: trainingConfig.validationSplit
                        },
                        trainingConfig,
                        speed
                    }
                });

                console.log("Starting training...");
                // Start training
                worker.postMessage({ type: 'start' });
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
                }
                set({ 
                    trainingState: "idle",
                    currentEpoch: 0,
                    currentBatch: 0
                });
                get().clearMetrics();
            },

            stepBatch: () => {
                const { worker, speed } = get();
                if (worker && speed === 0) {
                    worker.postMessage({ 
                        type: 'step', 
                        payload: { type: 'batch' } 
                    });
                }
            },

            stepEpoch: () => {
                const { worker, speed } = get();
                if (worker && speed === 0) {
                    worker.postMessage({ 
                        type: 'step', 
                        payload: { type: 'epoch' } 
                    });
                }
            },

            scrubToEpoch: (epoch) => {
                // For now, just update the display
                // In a full implementation, this would restore model state
                const { trainingMetrics } = get();
                const epochMetrics = trainingMetrics.filter(m => m.epoch <= epoch);
                console.log(`Scrubbing to epoch ${epoch}`, epochMetrics);
            },

            // Data management
            addMetric: (metric) => set(state => ({
                trainingMetrics: [...state.trainingMetrics, metric],
                currentEpoch: metric.epoch,
                currentBatch: metric.batch || 0
            })),

            updateActivations: (activations) => set({ layerActivations: activations }),
            updatePredictions: (predictions) => set({ modelPredictions: predictions }),

            clearMetrics: () => set({ trainingMetrics: [], layerActivations: [], modelPredictions: [] }),

            // Worker management
            initializeWorker: () => {
                const state = get();
                
                // Don't create a new worker if one already exists
                if (state.worker) {
                    return;
                }

                try {
                    const worker = new Worker(
                        new URL("../workers/training-worker.ts", import.meta.url),
                        { type: "module" }
                    );

                    worker.onmessage = (event) => {
                        const { type, payload } = event.data;
                        const state = get();

                        switch (type) {
                            case 'progress':
                                set({
                                    currentEpoch: payload.epoch,
                                    currentBatch: payload.batch
                                });
                                break;

                            case 'metrics':
                                state.addMetric({
                                    epoch: payload.epoch,
                                    loss: payload.loss,
                                    accuracy: payload.acc,
                                    valLoss: payload.val_loss,
                                    valAccuracy: payload.val_acc,
                                    timestamp: payload.timestamp
                                });
                                break;

                            case 'activations':
                                state.updateActivations(payload);
                                break;

                            case 'weights':
                                // Handle weight updates if needed
                                console.log('Weight update received', payload);
                                break;

                            case 'predictions':
                                console.log('Received predictions:', payload);
                                state.updatePredictions(payload);
                                break;

                            case 'complete':
                                set({ trainingState: "idle" });
                                break;

                            case 'paused':
                                set({ trainingState: "paused" });
                                break;

                            case 'error':
                                console.error('Training error:', payload.error);
                                set({ trainingState: "idle" });
                                break;
                        }
                    };

                    worker.onerror = (error) => {
                        console.error('Worker error:', error);
                        set({ trainingState: "idle" });
                    };

                    set({ worker });
                } catch (error) {
                    console.error('Failed to initialize worker:', error);
                }
            },

            disposeWorker: () => {
                const { worker } = get();
                if (worker) {
                    worker.terminate();
                    set({ worker: null });
                }
            },
        }),
        {
            name: "training-store",
            storage: createJSONStorage(() => localForage),
            partialize: (state) => ({
                trainingConfig: state.trainingConfig,
                speed: state.speed,
                // Don't persist worker or training state
            }),
        }
    )
);
