import { useRef, useEffect, useState, useCallback } from "react";
import { NetworkVisualization } from "../components/network-visualization";
import { ModelControls } from "../components/model-controls";
import { TrainingMetricsChart } from "../components/training-metrics-chart";
import { ShapeFixerDialog } from "../components/shape-fixer-dialog";
import { useModelStore } from "../lib/oscar-store";
import { useHyperparametersStore } from "../lib/oscar-store";
import { useDataStore } from "../lib/oscar-store";
import { TensorFlowModelMonitor } from "../lib/tensorflow-monitor";
import { ShapeValidator, type ShapeValidationResult, type ShapeFix } from "../lib/shape-validator";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";
import { Alert, AlertDescription } from "../components/ui/alert";
import { Loader2, Play, Square, Trash2 } from "lucide-react";
import * as tf from "@tensorflow/tfjs";

export default function ModelPage() {
    const canvasRef = useRef<HTMLDivElement>(null);
    const [monitor, setMonitor] = useState<TensorFlowModelMonitor | null>(null);
    const [isTrainingDemo, setIsTrainingDemo] = useState(false);
    const [demoModel, setDemoModel] = useState<tf.Sequential | null>(null);
    const [shapeValidationResult, setShapeValidationResult] = useState<ShapeValidationResult | null>(null);
    const [showShapeDialog, setShowShapeDialog] = useState(false);

    const { model, isTraining, viewMode, layers, resetModel } = useModelStore();
    const { dataset } = useDataStore();
    const {
        layers: hyperparametersLayers,
        learningRate,
        batchSize,
        epochs,
        optimizer,
        updateLayer
    } = useHyperparametersStore();

    // Debug hyperparameters
    useEffect(() => {
        console.log("Hyperparameters loaded:", {
            layersCount: hyperparametersLayers?.length || 0,
            firstLayer: hyperparametersLayers?.[0],
            optimizer,
            learningRate,
            batchSize,
            epochs
        });
    }, [hyperparametersLayers, optimizer, learningRate, batchSize, epochs]);

    // Function to convert activation string to TensorFlow activation
    const getActivationFunction = (activation: string): string => {
        switch (activation) {
            case 'ReLU':
                return 'relu';
            case 'Sigmoid':
                return 'sigmoid';
            case 'Tanh':
                return 'tanh';
            case 'Linear':
                return 'linear';
            case 'Custom':
                // For now, fallback to relu for custom functions
                // In a full implementation, you'd evaluate the custom function
                return 'relu';
            default:
                return 'relu';
        }
    };

    // Shape validation function
    const validateModelShapes = useCallback((): boolean => {
        if (!hyperparametersLayers || hyperparametersLayers.length === 0) {
            return true; // No layers to validate
        }

        // Get input shape from dataset or use default
        const inputShape = dataset ? ShapeValidator.getInputShapeFromData(dataset) : 784;

        // For now, assume regression (1 output) unless we have classification data
        const expectedOutputSize = 1; // This could be determined from dataset in the future

        // Validate the layer configuration
        const validationResult = ShapeValidator.validateModelShape(
            hyperparametersLayers,
            inputShape,
            expectedOutputSize
        );

        if (!validationResult.isValid) {
            setShapeValidationResult(validationResult);
            setShowShapeDialog(true);
            return false;
        }

        return true;
    }, [hyperparametersLayers, dataset]);

    // Handle applying shape fixes
    const handleApplyShapeFixes = useCallback((fixes: ShapeFix[]) => {
        const result = ShapeValidator.applySuggestionFixes(fixes);

        if (result.layers) {
            // Update all layers with the fixed configuration
            result.layers.forEach((layer) => {
                updateLayer(layer.id, layer);
            });
        }

        // Reset validation state
        setShapeValidationResult(null);
    }, [updateLayer]);

    // Handle dismissing shape validation
    const handleDismissShapeValidation = useCallback(() => {
        setShapeValidationResult(null);
    }, []);
    const buildModelFromHyperparameters = useCallback((): tf.Sequential => {
        console.log("Building model from hyperparameters:", {
            layersCount: hyperparametersLayers?.length || 0,
            layers: hyperparametersLayers?.map(l => ({ type: l.type, units: l.units, activation: l.activation }))
        });

        const model = tf.sequential();

        if (!hyperparametersLayers || hyperparametersLayers.length === 0) {
            console.log("No layers configured, creating default model");
            // Create a default model if no layers are configured
            model.add(tf.layers.dense({
                inputShape: [1],
                units: 10,
                activation: 'relu',
                name: 'default_layer_1'
            }));
            model.add(tf.layers.dense({
                units: 1,
                activation: 'linear',
                name: 'default_output'
            }));
            return model;
        }

        hyperparametersLayers.forEach((layer, index) => {
            console.log(`Adding layer ${index + 1}:`, layer);
            const layerConfig: any = {
                name: `layer_${index + 1}`
            };

            if (layer.type === 'Dense' && layer.units) {
                layerConfig.units = layer.units;
                if (layer.activation) {
                    layerConfig.activation = getActivationFunction(layer.activation);
                }

                // Add input shape for first layer
                if (index === 0) {
                    layerConfig.inputShape = [784]; // Default to MNIST-like input
                }

                model.add(tf.layers.dense(layerConfig));
            } else if (layer.type === 'Conv2D' && layer.filters && layer.kernelSize) {
                layerConfig.filters = layer.filters;
                layerConfig.kernelSize = layer.kernelSize;
                if (layer.activation) {
                    layerConfig.activation = getActivationFunction(layer.activation);
                }

                if (index === 0) {
                    layerConfig.inputShape = [28, 28, 1]; // Default for Conv2D
                }

                model.add(tf.layers.conv2d(layerConfig));
            } else if (layer.type === 'Conv1D' && layer.filters && layer.kernelSize) {
                layerConfig.filters = layer.filters;
                layerConfig.kernelSize = layer.kernelSize;
                if (layer.activation) {
                    layerConfig.activation = getActivationFunction(layer.activation);
                }

                if (index === 0) {
                    layerConfig.inputShape = [100, 1]; // Default for Conv1D
                }

                model.add(tf.layers.conv1d(layerConfig));
            } else if (layer.type === 'Flatten') {
                model.add(tf.layers.flatten(layerConfig));
            }
        });

        // Ensure the model has at least some layers
        if (model.layers.length === 0) {
            console.log("No valid layers found, adding fallback layer");
            model.add(tf.layers.dense({
                inputShape: [1],
                units: 10,
                activation: 'relu',
                name: 'fallback_layer'
            }));
        }

        console.log("Model building completed:", {
            totalLayers: model.layers.length,
            layerNames: model.layers.map(l => l.name)
        });

        return model;
    }, [hyperparametersLayers]);

    // Initialize monitor
    useEffect(() => {
        try {
            console.log("Initializing TensorFlow monitor...");
            const tfMonitor = new TensorFlowModelMonitor();
            setMonitor(tfMonitor);
            console.log("TensorFlow monitor initialized successfully");

            return () => {
                console.log("Disposing TensorFlow monitor...");
                tfMonitor.dispose();
            };
        } catch (error) {
            console.error("Failed to initialize TensorFlow monitor:", error);
        }
    }, []);

    // Component cleanup
    useEffect(() => {
        return () => {
            // Cleanup any remaining demo models on unmount
            setDemoModel(prevModel => {
                if (prevModel) {
                    try {
                        // Check if model is already disposed to avoid errors
                        prevModel.dispose();
                    } catch (error) {
                        // Silently handle disposal errors during cleanup
                        console.debug("Demo model disposal handled:", (error as Error).message);
                    }
                }
                return null;
            });
        };
    }, []);

    // Function to visualize model architecture without training
    const visualizeArchitecture = useCallback(async () => {
        if (!monitor) return;

        // Validate shapes before building the model
        if (!validateModelShapes()) {
            return; // Shape validation dialog is shown, stop here
        }

        try {
            // Dispose of previous demo model if it exists
            if (demoModel) {
                try {
                    demoModel.dispose();
                } catch (error) {
                    console.warn("Error disposing previous demo model:", error);
                }
            }

            // Small delay to ensure cleanup completes
            await new Promise(resolve => setTimeout(resolve, 100));

            const userModel = buildModelFromHyperparameters();

            console.log("Built model with layers:", userModel.layers.length);

            if (userModel.layers.length === 0) {
                console.error("Model has no layers!");
                return;
            }

            setDemoModel(userModel);

            // Initialize the model in the monitor for visualization
            monitor.initializeModel(userModel);

            // Generate sample data for initial visualization
            const firstLayer = userModel.layers[0];
            let inputShape: number[];

            if ('inputShape' in firstLayer && firstLayer.inputShape) {
                inputShape = firstLayer.inputShape as number[];
            } else if (firstLayer.batchInputShape) {
                inputShape = firstLayer.batchInputShape.slice(1).filter((x): x is number => x !== null);
            } else {
                inputShape = [1];
            }

            console.log("Using input shape:", inputShape);

            const sampleInputShape = [1, ...inputShape];
            const sampleInput = tf.randomUniform(sampleInputShape, -2, 2);
            await monitor.getLayerActivations(sampleInput);
            sampleInput.dispose();

            console.log("Architecture visualization completed");
        } catch (error) {
            console.error("Failed to visualize architecture:", error);
        }
    }, [monitor, buildModelFromHyperparameters, validateModelShapes]);

    // Auto-visualize architecture when component mounts or hyperparameters change
    useEffect(() => {
        console.log("Auto-visualization effect triggered:", {
            hasMonitor: !!monitor,
            isTrainingDemo,
            hasModel: !!model,
            layersCount: hyperparametersLayers?.length || 0
        });

        if (monitor && !isTrainingDemo && hyperparametersLayers && hyperparametersLayers.length > 0) {
            console.log("Starting auto-visualization...");
            visualizeArchitecture();
        }
    }, [monitor, hyperparametersLayers]);

    // Demo training function
    const startDemo = async () => {
        if (!monitor || isTrainingDemo) return;

        setIsTrainingDemo(true);

        try {
            // Create model from user's hyperparameters
            const userModel = buildModelFromHyperparameters();
            setDemoModel(userModel);

            // Initialize the model in the monitor for visualization
            monitor.initializeModel(userModel);

            // Generate sample data for initial visualization based on model input shape
            const firstLayer = userModel.layers[0];
            let inputShape: number[];

            // Try to get input shape from different sources
            if ('inputShape' in firstLayer && firstLayer.inputShape) {
                inputShape = firstLayer.inputShape as number[];
            } else if (firstLayer.batchInputShape) {
                inputShape = firstLayer.batchInputShape.slice(1).filter((x): x is number => x !== null);
            } else {
                // Default fallback
                inputShape = [1];
            }

            const sampleInputShape = [1, ...inputShape];
            const sampleInput = tf.randomUniform(sampleInputShape, -2, 2);
            await monitor.getLayerActivations(sampleInput);
            sampleInput.dispose();

            // Generate sample data for training
            const numSamples = 200;
            const xs = tf.randomUniform([numSamples, ...inputShape], -2, 2);
            const ys = tf.randomUniform([numSamples, 1], 0, 1); // Simple target data

            // Get optimizer string for monitor
            const optimizerStr = optimizer.toLowerCase() as "sgd" | "adam" | "rmsprop" | "adagrad";

            // Train the model with monitoring using user's hyperparameters
            await monitor.trainModel({
                model: userModel,
                dataset: { xs, ys },
                epochs: epochs,
                batchSize: typeof batchSize === 'number' ? batchSize : 32,
                learningRate: learningRate,
                optimizer: optimizerStr,
                onEpochEnd: (epoch, logs) => {
                    console.log(`Epoch ${epoch + 1}: loss = ${logs.loss}`);
                }
            });

            // Clean up tensors
            xs.dispose();
            ys.dispose();

        } catch (error) {
            console.error("Demo training failed:", error);
        } finally {
            setIsTrainingDemo(false);
        }
    };

    const stopDemo = () => {
        if (monitor) {
            monitor.stopMonitoring();
        }
        if (demoModel) {
            demoModel.dispose();
            setDemoModel(null);
        }
        setIsTrainingDemo(false);
    };

    return (
        <div className="w-full h-full flex flex-col p-6 gap-6 bg-background">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight">Neural Network Training</h1>
                    <p className="text-muted-foreground">
                        Train and visualize your custom neural network architecture configured in the hyperparameters tab
                    </p>
                </div>

                {/* Demo Controls */}
                <div className="flex gap-2">
                    {!isTrainingDemo ? (
                        <Button onClick={startDemo} className="flex items-center gap-2">
                            <Play className="h-4 w-4" />
                            Start Training
                        </Button>
                    ) : (
                        <Button onClick={stopDemo} variant="destructive" className="flex items-center gap-2">
                            <Square className="h-4 w-4" />
                            Stop Training
                        </Button>
                    )}

                    {(model || demoModel) && (
                        <Button
                            onClick={resetModel}
                            variant="outline"
                            className="flex items-center gap-2"
                        >
                            <Trash2 className="h-4 w-4" />
                            Clear Model
                        </Button>
                    )}
                </div>
            </div>

            {/* Status Alert */}
            {!model && !isTrainingDemo && (
                <Alert>
                    <AlertDescription>
                        Ready to train! Click "Start Training" to train the neural network configured in the hyperparameters tab.
                        Current model: {hyperparametersLayers.length} layers with {hyperparametersLayers.reduce((sum, layer) => sum + (layer.units || layer.filters || 0), 0)} total units.
                    </AlertDescription>
                </Alert>
            )}

            {isTrainingDemo && (
                <Alert>
                    <Loader2 className="h-4 w-4 animate-spin mr-2" />
                    <AlertDescription>
                        Training your custom neural network... Watch the layers learn in real-time!
                        Using {hyperparametersLayers.length} layers with {optimizer} optimizer.
                    </AlertDescription>
                </Alert>
            )}

            {/* Main Content */}
            <div className="flex-1 grid grid-cols-1 xl:grid-cols-3 gap-6">
                {/* Visualization Canvas - Takes up most space */}
                <div className="xl:col-span-2 flex flex-col gap-4">
                    {/* Controls */}
                    <ModelControls canvasRef={canvasRef as React.RefObject<HTMLElement>} />

                    {/* Network Visualization */}
                    <Card className="flex-1 min-h-[600px]">
                        <CardHeader>
                            <CardTitle className="flex items-center justify-between">
                                <span>Network Architecture</span>
                                <div className="text-sm text-muted-foreground">
                                    {viewMode === "layer" ? "Layer View" : "Neuron View"}
                                    {layers.length > 0 && ` • ${layers.length} layers`}
                                </div>
                            </CardTitle>
                            <CardDescription>
                                Interactive visualization of the neural network structure and activations.
                                {viewMode === "layer"
                                    ? " Each node represents a layer with color intensity showing activation magnitude."
                                    : " Each node represents an individual neuron with real-time activation values."
                                }
                            </CardDescription>
                        </CardHeader>
                        <CardContent className="p-0">
                            <div ref={canvasRef} className="w-full h-full min-h-[500px]">
                                <NetworkVisualization className="w-full h-full" />
                            </div>
                        </CardContent>
                    </Card>
                </div>

                {/* Side Panel */}
                <div className="flex flex-col gap-4">
                    <Tabs defaultValue="metrics" className="w-full">
                        <TabsList className="grid w-full grid-cols-2">
                            <TabsTrigger value="metrics">Training Metrics</TabsTrigger>
                            <TabsTrigger value="info">Model Info</TabsTrigger>
                        </TabsList>

                        <TabsContent value="metrics" className="space-y-4">
                            <TrainingMetricsChart />
                        </TabsContent>

                        <TabsContent value="info" className="space-y-4">
                            <Card>
                                <CardHeader>
                                    <CardTitle>Model Information</CardTitle>
                                </CardHeader>
                                <CardContent className="space-y-4">
                                    <div className="grid grid-cols-2 gap-4 text-sm">
                                        <div>
                                            <div className="font-medium">Status</div>
                                            <div className="text-muted-foreground">
                                                {isTraining ? "Training" : model ? "Loaded" : "Ready"}
                                            </div>
                                        </div>
                                        <div>
                                            <div className="font-medium">Layers</div>
                                            <div className="text-muted-foreground">
                                                {hyperparametersLayers.length}
                                            </div>
                                        </div>
                                        <div>
                                            <div className="font-medium">Optimizer</div>
                                            <div className="text-muted-foreground">
                                                {optimizer}
                                            </div>
                                        </div>
                                        <div>
                                            <div className="font-medium">Learning Rate</div>
                                            <div className="text-muted-foreground">
                                                {learningRate}
                                            </div>
                                        </div>
                                        <div>
                                            <div className="font-medium">Batch Size</div>
                                            <div className="text-muted-foreground">
                                                {batchSize}
                                            </div>
                                        </div>
                                        <div>
                                            <div className="font-medium">Epochs</div>
                                            <div className="text-muted-foreground">
                                                {epochs}
                                            </div>
                                        </div>
                                        <div>
                                            <div className="font-medium">View Mode</div>
                                            <div className="text-muted-foreground">
                                                {viewMode === "layer" ? "Layer" : "Neuron"}
                                            </div>
                                        </div>
                                        <div>
                                            <div className="font-medium">Backend</div>
                                            <div className="text-muted-foreground">
                                                TensorFlow.js
                                            </div>
                                        </div>
                                    </div>

                                    {hyperparametersLayers.length > 0 && (
                                        <div className="space-y-2">
                                            <div className="font-medium">Layer Configuration</div>
                                            <div className="space-y-1 text-xs">
                                                {hyperparametersLayers.map((layer, index) => (
                                                    <div key={layer.id} className="flex justify-between p-2 rounded bg-muted/50">
                                                        <span>Layer {index + 1}: {layer.type}</span>
                                                        <span className="text-muted-foreground">
                                                            {layer.type === 'Dense' && layer.units && `${layer.units} units`}
                                                            {layer.type === 'Conv2D' && layer.filters && `${layer.filters} filters`}
                                                            {layer.type === 'Conv1D' && layer.filters && `${layer.filters} filters`}
                                                            {layer.type === 'Flatten' && 'Flatten'}
                                                            {layer.activation && ` • ${layer.activation}`}
                                                        </span>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </CardContent>
                            </Card>
                        </TabsContent>
                    </Tabs>
                </div>
            </div>

            {/* Features Information */}
            <Card>
                <CardHeader>
                    <CardTitle>Visualization Features</CardTitle>
                    <CardDescription>
                        Interactive features available in the neural network visualizer
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
                        <div className="space-y-2">
                            <div className="font-medium">🎮 Interactive Controls</div>
                            <ul className="space-y-1 text-muted-foreground">
                                <li>• Play/Pause live updates</li>
                                <li>• Switch between layer/neuron view</li>
                                <li>• Focus on specific layers</li>
                                <li>• Zoom and pan controls</li>
                            </ul>
                        </div>
                        <div className="space-y-2">
                            <div className="font-medium">📊 Real-time Metrics</div>
                            <ul className="space-y-1 text-muted-foreground">
                                <li>• Live training loss</li>
                                <li>• Accuracy tracking</li>
                                <li>• Layer activation statistics</li>
                                <li>• Weight magnitude visualization</li>
                            </ul>
                        </div>
                        <div className="space-y-2">
                            <div className="font-medium">📸 Snapshots</div>
                            <ul className="space-y-1 text-muted-foreground">
                                <li>• Capture network state</li>
                                <li>• Export as PNG images</li>
                                <li>• Compare different epochs</li>
                                <li>• Training progress history</li>
                            </ul>
                        </div>
                        <div className="space-y-2">
                            <div className="font-medium">🎨 Visual Features</div>
                            <ul className="space-y-1 text-muted-foreground">
                                <li>• Color-coded activations</li>
                                <li>• Edge thickness for weights</li>
                                <li>• Hover tooltips with stats</li>
                                <li>• Responsive layout</li>
                            </ul>
                        </div>
                    </div>
                </CardContent>
            </Card>

            {/* Shape Fixer Dialog */}
            {shapeValidationResult && (
                <ShapeFixerDialog
                    open={showShapeDialog}
                    onOpenChange={setShowShapeDialog}
                    validationResult={shapeValidationResult}
                    onApplyFixes={handleApplyShapeFixes}
                    onDismiss={handleDismissShapeValidation}
                />
            )}
        </div>
    );
}