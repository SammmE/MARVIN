import * as tf from "@tensorflow/tfjs";

// Types for worker communication
interface TrainingConfig {
    modelConfig: any;
    dataConfig: {
        xs: number[][];
        ys: number[][];
        validationSplit?: number;
    };
    trainingConfig: {
        epochs: number;
        batchSize: number;
        learningRate: number;
        optimizer: string;
    };
    speed: number; // 0 = manual, 1 = real-time, >1 = accelerated
}

interface TrainingMessage {
    type: 'start' | 'pause' | 'resume' | 'stop' | 'step' | 'config';
    payload?: any;
}

interface TrainingResponse {
    type: 'progress' | 'metrics' | 'activations' | 'weights' | 'complete' | 'error' | 'paused' | 'predictions';
    payload: any;
}

// Worker state
let model: tf.Sequential | null = null;
let isTraining = false;
let isPaused = false;
let currentEpoch = 0;
let currentBatch = 0;
let trainingData: { xs: tf.Tensor; ys: tf.Tensor } | null = null;
let validationData: { xs: tf.Tensor; ys: tf.Tensor } | null = null;
let config: TrainingConfig | null = null;

// Training loop with frame yielding
async function trainStep(xs: tf.Tensor, ys: tf.Tensor, batchSize: number): Promise<any> {
    if (!model) throw new Error("Model not initialized");
    
    const history = await model.fit(xs, ys, {
        batchSize,
        epochs: 1,
        validationData: validationData ? [validationData.xs, validationData.ys] : undefined,
        callbacks: {
            onBatchEnd: async (batch, logs) => {
                currentBatch = batch;
                
                // Send progress update
                postMessage({
                    type: 'progress',
                    payload: {
                        epoch: currentEpoch,
                        batch,
                        metrics: logs
                    }
                } as TrainingResponse);

                // Skip activation extraction during training to avoid disposal issues
                // The sendActivations functionality can cause layer disposal conflicts
                // TODO: Implement safer activation extraction or disable during active training
                // if (batch % 10 === 0 && isTraining && !isPaused && model) {
                //     try {
                //         await sendActivations(xs.slice([0, 0], [1, -1]));
                //     } catch (activationError) {
                //         console.warn('Error sending activations:', activationError);
                //     }
                // }

                // Yield to main thread
                await tf.nextFrame();
                
                // Check if paused or stopped
                if (isPaused || !isTraining) {
                    throw new Error("Training paused or stopped");
                }
            },
            onEpochEnd: async (epoch, logs) => {
                currentEpoch = epoch;
                
                // Send metrics update with predictions
                postMessage({
                    type: 'metrics',
                    payload: {
                        epoch,
                        ...logs,
                        timestamp: Date.now()
                    }
                } as TrainingResponse);

                // Send model predictions for visualization
                if (model && trainingData) {
                    try {
                        await sendPredictions();
                    } catch (predictionError) {
                        console.warn('Error sending predictions:', predictionError);
                    }
                }

                // Skip weight updates during training to avoid disposal issues
                // if (model) {
                //     try {
                //         await sendWeights();
                //     } catch (weightError) {
                //         console.warn('Error sending weights:', weightError);
                //     }
                // }

                // Yield to main thread
                await tf.nextFrame();
            }
        }
    });

    return history;
}

async function sendPredictions() {
    if (!model || !trainingData) return;

    try {
        // Get predictions for the training data
        const predictions = model.predict(trainingData.xs) as tf.Tensor;
        const predictionArray = await predictions.data();
        
        // Get the actual number of samples in the training data
        const numSamples = trainingData.xs.shape[0];
        console.log('sendPredictions: numSamples =', numSamples, 'predictionArray.length =', predictionArray.length);
        
        // Convert to array of prediction objects, but only for the actual number of samples
        const predictionsData = [];
        for (let index = 0; index < numSamples; index++) {
            const prediction = predictionArray[index];
            
            const inputSlice = trainingData!.xs.slice([index, 0], [1, -1]);
            const actualSlice = trainingData!.ys.slice([index, 0], [1, -1]);
            
            const input = Array.from(inputSlice.dataSync())[0];
            const actual = Array.from(actualSlice.dataSync())[0];
            
            inputSlice.dispose();
            actualSlice.dispose();
            
            predictionsData.push({
                index,
                prediction,
                input,
                actual
            });
        }

        console.log('sendPredictions: sending', predictionsData.length, 'predictions');
        postMessage({
            type: 'predictions',
            payload: predictionsData
        } as TrainingResponse);

        predictions.dispose();
    } catch (error) {
        console.warn('Error in sendPredictions:', error);
    }
}

async function sendActivations(inputSample: tf.Tensor) {
    // Check if model exists
    if (!model) return;

    const layerActivations: any[] = [];
    
    try {
        // Get activations from each layer
        for (let i = 0; i < model.layers.length; i++) {
            const layer = model.layers[i];
            
            // Create a temporary model to get layer activations
            let layerModel: tf.LayersModel | null = null;
            let activation: tf.Tensor | null = null;
            
            try {
                layerModel = tf.model({
                    inputs: model.input,
                    outputs: layer.output
                });
                
                activation = layerModel.predict(inputSample) as tf.Tensor;
                const activationArray = await activation.data();
                
                layerActivations.push({
                    layerId: `layer-${i}`,
                    layerName: layer.name || `Layer ${i + 1}`,
                    activations: Array.from(activationArray),
                    gradients: [], // Would need custom training loop for gradients
                });
            } catch (error) {
                console.warn(`Failed to extract activations for layer ${i}:`, error);
                // Continue with next layer
            } finally {
                // Clean up resources
                if (activation && !activation.isDisposed) {
                    activation.dispose();
                }
                if (layerModel) {
                    try {
                        layerModel.dispose();
                    } catch (disposeError) {
                        console.warn('Error disposing layerModel:', disposeError);
                    }
                }
            }
        }

        if (layerActivations.length > 0) {
            postMessage({
                type: 'activations',
                payload: layerActivations
            } as TrainingResponse);
        }
    } catch (error) {
        console.warn('Error in sendActivations:', error);
    }
}

async function sendWeights() {
    if (!model) return;

    try {
        const weightsData: any[] = [];
        
        model.layers.forEach((layer, index) => {
            try {
                const weights = layer.getWeights();
                if (weights.length > 0) {
                    const weightArrays = weights.map(w => Array.from(w.dataSync()));
                    weightsData.push({
                        layerId: `layer-${index}`,
                        weights: weightArrays[0], // Main weights
                        biases: weightArrays[1] || null, // Biases if present
                    });
                }
            } catch (layerError) {
                console.warn(`Error getting weights for layer ${index}:`, layerError);
            }
        });

        if (weightsData.length > 0) {
            postMessage({
                type: 'weights',
                payload: weightsData
            } as TrainingResponse);
        }
    } catch (error) {
        console.warn('Error in sendWeights:', error);
    }
}

async function createModel(modelConfig: any): Promise<tf.Sequential> {
    const model = tf.sequential();
    
    modelConfig.layers.forEach((layerConfig: any, index: number) => {
        if (layerConfig.type === 'dense') {
            const config: any = {
                units: layerConfig.units,
                activation: layerConfig.activation,
                name: `layer_${index + 1}`
            };
            
            if (index === 0) {
                config.inputShape = layerConfig.inputShape;
            }
            
            model.add(tf.layers.dense(config));
        }
    });

    // Compile model
    let optimizer;
    const optimizerType = (modelConfig.optimizer || 'adam').toLowerCase();
    switch (optimizerType) {
        case 'sgd':
            optimizer = tf.train.sgd(modelConfig.learningRate);
            break;
        case 'rmsprop':
            optimizer = tf.train.rmsprop(modelConfig.learningRate);
            break;
        case 'adagrad':
            optimizer = tf.train.adagrad(modelConfig.learningRate);
            break;
        case 'adam':
        default:
            optimizer = tf.train.adam(modelConfig.learningRate);
            break;
    }
    
    model.compile({
        optimizer,
        loss: modelConfig.loss || 'meanSquaredError',
        metrics: modelConfig.metrics || ['mae']
    });

    return model;
}

async function startTraining() {
    if (!config || !trainingData) {
        throw new Error("Training not configured");
    }

    console.log("Starting training with config:", {
        modelExists: !!model,
        trainingDataExists: !!trainingData,
        configExists: !!config
    });

    isTraining = true;
    isPaused = false;

    try {
        // Create model if not exists or if it was disposed
        if (!model) {
            console.log("Creating new model...");
            model = await createModel(config.modelConfig);
            console.log("Model created successfully");
            
            // Send initial predictions to show the untrained model
            try {
                await sendPredictions();
            } catch (predictionError) {
                console.warn('Error sending initial predictions:', predictionError);
            }
        }

        const { epochs, batchSize } = config.trainingConfig;
        
        for (let epoch = 0; epoch < epochs && isTraining && !isPaused; epoch++) {
            currentEpoch = epoch;
            
            // Apply speed control
            if (config?.speed === 0) {
                // Manual mode - wait for step command
                isPaused = true;
                postMessage({
                    type: 'paused',
                    payload: { epoch, reason: 'manual_mode' }
                } as TrainingResponse);
                return;
            } else if (config && config.speed < 1) {
                // Slower than real-time
                const delay = (1 - config.speed) * 1000;
                await new Promise(resolve => setTimeout(resolve, delay));
            }
            // For speed > 1, training runs as fast as possible
            
            await trainStep(trainingData.xs, trainingData.ys, batchSize);
            
            // Yield between epochs
            await tf.nextFrame();
        }

        if (isTraining) {
            postMessage({
                type: 'complete',
                payload: { finalEpoch: currentEpoch }
            } as TrainingResponse);
        }

    } catch (error) {
        if (isPaused) {
            postMessage({
                type: 'paused',
                payload: { epoch: currentEpoch }
            } as TrainingResponse);
        } else {
            postMessage({
                type: 'error',
                payload: { error: error instanceof Error ? error.message : String(error) }
            } as TrainingResponse);
        }
    }
}

// Message handler
self.onmessage = async (event: MessageEvent<TrainingMessage>) => {
    const { type, payload } = event.data;
    
    console.log("Worker received message:", type, payload ? "with payload" : "no payload");

    try {
        switch (type) {
            case 'config':
                console.log("Worker received config, current model exists:", !!model);
                
                // Simply set the new config - don't try to dispose existing model
                // since we're using a fresh worker instance
                config = payload;
                
                // Prepare training data
                if (config) {
                    console.log("Preparing training data...");
                    const { xs: xsData, ys: ysData, validationSplit } = config.dataConfig;
                    const xs = tf.tensor2d(xsData);
                    const ys = tf.tensor2d(ysData);
                    
                    if (validationSplit && validationSplit > 0) {
                        const splitIndex = Math.floor(xsData.length * (1 - validationSplit));
                        trainingData = {
                            xs: xs.slice([0, 0], [splitIndex, -1]),
                            ys: ys.slice([0, 0], [splitIndex, -1])
                        };
                        validationData = {
                            xs: xs.slice([splitIndex, 0], [-1, -1]),
                            ys: ys.slice([splitIndex, 0], [-1, -1])
                        };
                    } else {
                        trainingData = { xs, ys };
                    }
                    
                    // Dispose the original tensors since we've sliced them
                    xs.dispose();
                    ys.dispose();
                    console.log("Training data prepared successfully");
                }
                break;

            case 'start':
                await startTraining();
                break;

            case 'pause':
                isPaused = true;
                break;

            case 'resume':
                if (isPaused) {
                    isPaused = false;
                    await startTraining();
                }
                break;

            case 'stop':
                isTraining = false;
                isPaused = false;
                currentEpoch = 0;
                currentBatch = 0;
                
                // Clean disposal since we're using fresh workers
                if (model) {
                    try {
                        model.dispose();
                    } catch (disposeError) {
                        console.warn('Error disposing model on stop:', disposeError);
                    }
                    model = null;
                }
                if (trainingData) {
                    try {
                        trainingData.xs.dispose();
                        trainingData.ys.dispose();
                    } catch (disposeError) {
                        console.warn('Error disposing training data on stop:', disposeError);
                    }
                    trainingData = null;
                }
                if (validationData) {
                    try {
                        validationData.xs.dispose();
                        validationData.ys.dispose();
                    } catch (disposeError) {
                        console.warn('Error disposing validation data on stop:', disposeError);
                    }
                    validationData = null;
                }
                break;

            case 'step':
                if (config && trainingData && config.speed === 0) {
                    // Manual step in manual mode
                    if (payload.type === 'batch') {
                        // Step one batch
                        await trainStep(trainingData.xs, trainingData.ys, config.trainingConfig.batchSize);
                    } else if (payload.type === 'epoch') {
                        // Step one epoch
                        currentEpoch++;
                        await trainStep(trainingData.xs, trainingData.ys, config.trainingConfig.batchSize);
                    }
                }
                break;
        }
    } catch (error) {
        postMessage({
            type: 'error',
            payload: { error: error instanceof Error ? error.message : String(error) }
        } as TrainingResponse);
    }
};
