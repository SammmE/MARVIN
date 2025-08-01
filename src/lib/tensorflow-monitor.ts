import * as tf from "@tensorflow/tfjs";
import { useOscarStore, type LayerVisualization } from "./oscar-store";
import React from "react";

export interface ModelTrainingConfig {
	model: tf.Sequential;
	dataset: {
		xs: tf.Tensor;
		ys: tf.Tensor;
	};
	validationData?: {
		xs: tf.Tensor;
		ys: tf.Tensor;
	};
	epochs: number;
	batchSize: number;
	learningRate: number;
	optimizer: "adam" | "rmsprop" | "adagrad" | "sgd";
	onEpochEnd?: (epoch: number, logs: tf.Logs) => void;
	onBatchEnd?: (batch: number, logs: tf.Logs) => void;
}

export class TensorFlowModelMonitor {
	private storeActions: ReturnType<typeof useOscarStore.getState>;
	private updateIntervalId: number | null = null;
	private isMonitoring = false;
	private isGettingActivations = false;
	private currentModel: tf.Sequential | null = null;

	constructor() {
		// Get store actions
		this.storeActions = useOscarStore.getState();
	}

	// Initialize the monitor with a model
	initializeModel(model: tf.Sequential): void {
		console.log("TensorFlowModelMonitor.initializeModel called with model:", {
			layerCount: model.layers.length,
			layerNames: model.layers.map(l => l.name),
			layerTypes: model.layers.map(l => l.getClassName())
		});

		// Get fresh store reference each time
		this.storeActions = useOscarStore.getState();
		
		// Don't clear previous layers on initialization - preserve them for visualization
		// This allows the network visualization to continue showing layers even during monitor transitions
		
		// Store reference to current model
		this.currentModel = model;
		
		this.storeActions.setModel(model);
		console.log("Model set in store");
		
		this.storeActions.initializeVisualization();
		console.log("Visualization initialized");
		
		this.extractModelLayers(model);
		console.log("Model layers extracted");
		
		// Trigger a visualization update to ensure nodes and edges are generated
		this.storeActions.updateVisualizationData();
		console.log("Visualization data updated");
	}

	// Extract layer information from TensorFlow model
	private extractModelLayers(model: tf.Sequential): void {
		console.log("Extracting model layers:", model.layers.length);
		
		// First, create the layer structure
		const layerVisualizations: LayerVisualization[] = [];
		
		model.layers.forEach((layer: tf.layers.Layer, index: number) => {
			const layerId = `layer-${index}`;
			console.log(`Processing layer ${index}:`, {
				id: layerId,
				name: layer.name,
				type: layer.getClassName(),
				outputShape: layer.outputShape
			});

			// Create initial layer visualization object
			const layerViz: LayerVisualization = {
				id: layerId,
				name: layer.name,
				type: layer.getClassName(),
				activations: [],
				shape: layer.outputShape as number[],
				position: { x: index * 200, y: 50 },
				isInput: index === 0,
				isOutput: index === model.layers.length - 1,
				stats: {
					mean: 0,
					max: 0,
					min: 0,
					std: 0
				}
			};

			layerVisualizations.push(layerViz);

			// Extract weights if available, with safety check for disposed layers
			try {
				const weights = layer.getWeights();
				if (weights.length > 0) {
					console.log(`Layer ${index} has ${weights.length} weight tensors`);
					const weightMatrix = weights[0].arraySync() as number[][];
					const biases =
						weights.length > 1 ? (weights[1].arraySync() as number[]) : undefined;
					
					// Update the layer with weights
					layerViz.weights = weightMatrix;
					layerViz.biases = biases;
				} else {
					console.log(`Layer ${index} has no weights yet`);
				}
			} catch (error) {
				console.warn(`Failed to extract weights for layer ${index}:`, error);
				// Continue processing other layers
			}
		});
		
		// Set all layers at once
		this.storeActions.setLayers(layerVisualizations);
		
		console.log("Model layer extraction completed");
	}

	// Start monitoring during training
	startMonitoring(): void {
		if (this.isMonitoring) return;

		this.isMonitoring = true;
		const store = useOscarStore.getState();
		const { updateInterval, liveUpdateEnabled } = store;

		if (liveUpdateEnabled) {
			this.updateIntervalId = window.setInterval(() => {
				this.updateVisualization();
			}, updateInterval);
		}
	}

	// Stop monitoring
	stopMonitoring(): void {
		this.isMonitoring = false;
		if (this.updateIntervalId) {
			clearInterval(this.updateIntervalId);
			this.updateIntervalId = null;
		}
	}

	// Update visualization with current model state
	private updateVisualization(): void {
		const store = useOscarStore.getState();
		const { model } = store;
		if (!model) return;

		// Extract current weights and activations
		model.layers.forEach((layer: tf.layers.Layer, index: number) => {
			const layerId = `layer-${index}`;

			// Get current weights
			const weights = layer.getWeights();
			if (weights.length > 0) {
				const weightMatrix = weights[0].arraySync() as number[][];
				const biases =
					weights.length > 1 ? (weights[1].arraySync() as number[]) : undefined;
				this.storeActions.updateLayerWeights(layerId, weightMatrix, biases);
			}
		});
	}

	// Helper method to safely check if a layer is disposed
	private isLayerDisposed(layer: tf.layers.Layer): boolean {
		try {
			// Try to access a property that would throw if disposed
			layer.name; // Just access the property without assigning
			return false;
		} catch (error) {
			const errorMessage = (error as Error).message;
			return errorMessage.includes('disposed') || errorMessage.includes('already disposed');
		}
	}

	// Helper method to safely check if a model is disposed
	private isModelDisposed(model: tf.Sequential): boolean {
		try {
			// Try to access model properties that would throw if disposed
			model.layers.length; // Just access the property without assigning
			return false;
		} catch (error) {
			const errorMessage = (error as Error).message;
			return errorMessage.includes('disposed') || errorMessage.includes('already disposed');
		}
	}

	// Get activations for a given input
	async getLayerActivations(input: tf.Tensor): Promise<void> {
		// Prevent concurrent calls
		if (this.isGettingActivations) {
			console.log("Already getting activations, skipping...");
			return;
		}

		this.isGettingActivations = true;

		try {
			const store = useOscarStore.getState();
			const { model } = store;
			if (!model) {
				console.log("No model in store, skipping activation retrieval");
				return;
			}

			// Check if this is still the same model we initialized with
			if (model !== this.currentModel) {
				console.warn("Model has changed since initialization, skipping activation retrieval");
				return;
			}

			// Check if model is disposed before proceeding
			if (this.isModelDisposed(model)) {
				console.warn("Model is disposed, skipping activation retrieval");
				return;
			}

			// Try to access model layers to check if model is still valid
			if (!model.layers || model.layers.length === 0) {
				console.warn("Model has no layers, skipping activation retrieval");
				return;
			}

			// First, compile the model if not already compiled
			if (!model.optimizer) {
				try {
					model.compile({
						optimizer: 'adam',
						loss: 'meanSquaredError'
					});
				} catch (compileError) {
					console.warn("Could not compile model, it may be disposed:", compileError);
					return;
				}
			}

			// Get predictions for each layer
			for (let i = 0; i < model.layers.length; i++) {
				const layerId = `layer-${i}`;

				try {
					// Double-check model hasn't been disposed during the loop
					if (this.isModelDisposed(model)) {
						console.warn(`Model was disposed during layer iteration at layer ${i}, stopping`);
						break;
					}

					// Check if the layer still exists and is valid
					const layer = model.layers[i];
					if (!layer || !layer.output) {
						console.warn(`Layer ${i} is invalid, skipping activation`);
						continue;
					}

					// Check if the specific layer is disposed
					if (this.isLayerDisposed(layer)) {
						console.warn(`Layer ${i} is disposed, skipping activation`);
						continue;
					}

					// Create intermediate model to get layer output
					const intermediateModel = tf.model({
						inputs: model.inputs,
						outputs: layer.output,
					});

					const activation = intermediateModel.predict(input) as tf.Tensor;
					
					// Handle different tensor shapes
					let activationData: number[][];
					const shape = activation.shape;
					
					if (shape.length === 2) {
						// Batch dimension + features, flatten to 2D
						activationData = activation.arraySync() as number[][];
					} else if (shape.length === 1) {
						// Just features, wrap in array
						activationData = [activation.arraySync() as number[]];
					} else {
						// Flatten other shapes
						const flattened = activation.flatten();
						activationData = [flattened.arraySync() as number[]];
						flattened.dispose();
					}

					this.storeActions.updateLayerActivations(layerId, activationData);

					activation.dispose();
					intermediateModel.dispose();
				} catch (error) {
					const errorMessage = (error as Error).message;
					
					// Check if this is a disposal error
					if (errorMessage.includes('disposed') || errorMessage.includes('already disposed')) {
						console.warn(`Layer ${i} was disposed during activation retrieval, stopping layer iteration`);
						// If any layer is disposed, likely the whole model is disposed, so break
						break;
					}
					
					console.warn(`Could not get activations for layer ${i}:`, error);
					// Create dummy activation data as fallback for non-disposal errors
					try {
						const layer = model.layers[i];
						if (layer && !this.isLayerDisposed(layer)) {
							const outputShape = layer.outputShape as number[];
							const dummyData = outputShape.length > 1 
								? [new Array(outputShape[outputShape.length - 1]).fill(0)]
								: [[0]];
							this.storeActions.updateLayerActivations(layerId, dummyData);
						}
					} catch (shapeError) {
						// If we can't even get the output shape, skip this layer
						console.warn(`Could not get shape for layer ${i}, skipping:`, shapeError);
					}
				}
			}
		} catch (error) {
			const errorMessage = (error as Error).message;
			
			// Check if this is a disposal error at the model level
			if (errorMessage.includes('disposed') || errorMessage.includes('already disposed')) {
				console.warn("Model was disposed during activation retrieval, operation aborted");
			} else {
				console.error("Error getting layer activations:", error);
			}
		} finally {
			this.isGettingActivations = false;
		}
		
		// Always trigger visualization update after getting activations
		this.storeActions.updateVisualizationData();
	}

	// Train model with live monitoring
	async trainModel(config: ModelTrainingConfig): Promise<tf.History> {
		const {
			model,
			dataset,
			validationData,
			epochs,
			batchSize,
			learningRate,
			optimizer,
			onEpochEnd,
			onBatchEnd,
		} = config;

		// Set up optimizer
		let opt: tf.Optimizer;
		switch (optimizer) {
			case "sgd":
				opt = tf.train.sgd(learningRate);
				break;
			case "adam":
				opt = tf.train.adam(learningRate);
				break;
			case "rmsprop":
				opt = tf.train.rmsprop(learningRate);
				break;
			case "adagrad":
				opt = tf.train.adagrad(learningRate);
				break;
			default:
				opt = tf.train.adam(learningRate);
		}

		// Compile model
		model.compile({
			optimizer: opt,
			loss: "meanSquaredError",
			metrics: ["accuracy"],
		});

		// Initialize monitoring
		this.initializeModel(model);
		this.storeActions.setTrainingState(true);
		this.storeActions.setTotalEpochs(epochs);
		this.storeActions.clearTrainingMetrics();

		this.startMonitoring();

		try {
			// Custom training loop for better control
			const history = await model.fit(dataset.xs, dataset.ys, {
				epochs,
				batchSize,
				validationData: validationData
					? [validationData.xs, validationData.ys]
					: undefined,
				shuffle: true,
				callbacks: {
					onEpochBegin: async (epoch) => {
						this.storeActions.setCurrentEpoch(epoch + 1);
					},
					onEpochEnd: async (epoch, logs) => {
						// Update training metrics
						this.storeActions.addTrainingMetric({
							epoch: epoch + 1,
							loss: logs?.loss || 0,
							accuracy: logs?.acc || logs?.accuracy,
							valLoss: logs?.val_loss,
							valAccuracy: logs?.val_acc || logs?.val_accuracy,
							timestamp: Date.now(),
						});

						// Update layer activations with a sample input
						if (dataset.xs.shape[0] > 0) {
							const sampleInput = dataset.xs.slice([0, 0], [1, -1]);
							await this.getLayerActivations(sampleInput);
							sampleInput.dispose();
						}

						// Call custom epoch end callback
						if (onEpochEnd) {
							onEpochEnd(epoch, logs || {});
						}
					},
					onBatchEnd: async (batch, logs) => {
						// Update visualization periodically during training
						if (batch % 10 === 0) {
							this.updateVisualization();
						}

						if (onBatchEnd) {
							onBatchEnd(batch, logs || {});
						}
					},
				},
			});

			return history;
		} finally {
			this.stopMonitoring();
			this.storeActions.setTrainingState(false);
		}
	}

	// Predict and visualize activations
	async predict(input: tf.Tensor): Promise<tf.Tensor> {
		const store = useOscarStore.getState();
		const { model } = store;
		if (!model) {
			throw new Error("No model available for prediction");
		}

		// Get layer activations for the input
		await this.getLayerActivations(input);

		// Make prediction
		return model.predict(input) as tf.Tensor;
	}

	// Create sample data for testing
	static createSampleData(numSamples: number = 100): {
		xs: tf.Tensor;
		ys: tf.Tensor;
	} {
		// Generate simple synthetic data (sine wave)
		const xs = tf.randomUniform([numSamples, 1], -Math.PI, Math.PI);
		const ys = xs.sin().add(tf.randomNormal([numSamples, 1], 0, 0.1));

		return { xs, ys };
	}

	// Create a simple model for testing
	static createSampleModel(
		inputDim: number = 1,
		hiddenUnits: number[] = [10, 5],
	): tf.Sequential {
		const model = tf.sequential();

		// Input layer
		model.add(
			tf.layers.dense({
				inputShape: [inputDim],
				units: hiddenUnits[0],
				activation: "relu",
				name: "hidden_1",
			}),
		);

		// Hidden layers
		for (let i = 1; i < hiddenUnits.length; i++) {
			model.add(
				tf.layers.dense({
					units: hiddenUnits[i],
					activation: "relu",
					name: `hidden_${i + 1}`,
				}),
			);
		}

		// Output layer
		model.add(
			tf.layers.dense({
				units: 1,
				activation: "linear",
				name: "output",
			}),
		);

		return model;
	}

	// Dispose of resources
	dispose(): void {
		console.log("TensorFlowModelMonitor.dispose called - preserving layer visualizations");
		this.stopMonitoring();
		
		// Clear the current model reference
		this.currentModel = null;
		
		const store = useOscarStore.getState();
		const { model } = store;
		if (model) {
			try {
				// Check if model is already disposed before attempting disposal
				// TensorFlow models don't have a clean way to check disposal state,
				// so we just handle the error gracefully
				model.dispose();
				console.log("Model disposed successfully");
			} catch (error) {
				// Silently handle disposal errors as they're common during cleanup
				console.debug("Model disposal handled:", (error as Error).message);
			}
			
			// Only clear the model reference, preserve layers for visualization
			this.storeActions.setModel(null);
			console.log("Model reference cleared, layer visualizations preserved");
		}
	}
}

// Hook for easy integration with React components
export const useTensorFlowMonitor = () => {
	const monitor = React.useMemo(() => new TensorFlowModelMonitor(), []);

	React.useEffect(() => {
		return () => {
			monitor.dispose();
		};
	}, [monitor]);

	return monitor;
};
