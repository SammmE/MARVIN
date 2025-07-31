import React, { useEffect, useState, useMemo } from "react";
import { TrainingControls } from "../components/training-controls";
import { LiveMetricsChart } from "../components/live-metrics-chart";
import { ActivationPanelGrid } from "../components/activation-mini-panels";
import { TrainingDataVisualization } from "../components/training-data-visualization";
import { useTrainingStore } from "../lib/training-store";
import { useDataStore } from "../lib/data-store";
import { useHyperparametersStore } from "../lib/hyperparameters-store";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";
import { Alert, AlertDescription } from "../components/ui/alert";
import { Button } from "../components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "../components/ui/dialog";
import { AlertCircle, Brain, BarChart3, Activity, Wrench } from "lucide-react";

// Generate sample training data
const generateSampleData = (problemType: "regression" | "classification", numPoints: number = 100) => {
    const data = [];
    for (let i = 0; i < numPoints; i++) {
        const x = (Math.random() - 0.5) * 10; // Random x between -5 and 5
        let y;

        if (problemType === "regression") {
            // Simple quadratic relationship with noise
            y = 0.5 * x * x + 2 * x + 1 + (Math.random() - 0.5) * 2;
        } else {
            // Classification: sigmoid-like boundary
            const prob = 1 / (1 + Math.exp(-(x + Math.sin(x) * 0.5)));
            y = Math.random() < prob ? 1 : 0;
        }

        data.push({ x, y });
    }
    return data;
};

export default function TrainingPage() {
    const {
        trainingState,
        speed,
        currentEpoch,
        totalEpochs,
        currentBatch,
        totalBatches,
        trainingMetrics,
        layerActivations,
        modelPredictions,
        trainingConfig,
        dataset,
        worker,
        lastError,
        errorType,

        // Actions
        setSpeed,
        startTraining,
        pauseTraining,
        resumeTraining,
        stopTraining,
        stepBatch,
        stepEpoch,
        scrubToEpoch,
        setDataset,
        setTrainingConfig,
        setModelConfig,
        clearError,
        initializeWorker,
        disposeWorker,
    } = useTrainingStore();

    // Get data from the data store
    const dataStore = useDataStore();

    // Get hyperparameters from the hyperparameters store
    const {
        epochs,
        learningRate,
        optimizer,
        batchSize,
        customBatchSize,
        layers,
        updateLayer,
    } = useHyperparametersStore();

    const [problemType, setProblemType] = useState<"regression" | "classification">("regression");
    const [showErrorDialog, setShowErrorDialog] = useState(false);

    // Show error dialog when an error occurs
    useEffect(() => {
        if (lastError && errorType) {
            setShowErrorDialog(true);
        }
    }, [lastError, errorType]);

    // Autofix function for shape mismatches
    const handleAutofix = () => {
        if (errorType === 'shape_mismatch' && dataset) {
            console.log('Applying autofix for shape mismatch...');

            // Extract expected and actual shapes from error message
            const errorMessage = lastError || '';

            // Parse the error to understand the shape mismatch
            let targetShape = 1; // Default to 1 for regression
            if (problemType === "classification") {
                // For classification, determine number of unique classes
                const uniqueClasses = new Set(dataset.ys.flat());
                targetShape = uniqueClasses.size;
            }

            // Update the last layer of the model to match the target shape
            const updatedLayers = [...layers];
            if (updatedLayers.length > 0) {
                const lastLayerIndex = updatedLayers.length - 1;
                const lastLayer = { ...updatedLayers[lastLayerIndex] };
                lastLayer.units = targetShape;
                lastLayer.activation = problemType === "classification" ?
                    (targetShape > 2 ? "Softmax" : "Sigmoid") : "ReLU";

                // Update the hyperparameters store
                updateLayer(lastLayer.id, {
                    units: targetShape,
                    activation: lastLayer.activation
                });

                updatedLayers[lastLayerIndex] = lastLayer;

                console.log('Autofix: Updated last layer in hyperparameters store:', {
                    layerId: lastLayer.id,
                    units: targetShape,
                    activation: lastLayer.activation,
                    problemType
                });

                // Update the hyperparameters store with the fixed layers
                // Note: We need to access the setter from hyperparameters store
                // For now, let's create a new model config directly
                const fixedModelConfig = {
                    layers: updatedLayers.map((layer, index) => {
                        const layerConfig: any = {
                            type: layer.type.toLowerCase(),
                        };

                        if (layer.type === "Dense") {
                            layerConfig.units = layer.units || 64;
                            layerConfig.activation = layer.activation?.toLowerCase() || 'relu';

                            // Add input shape for first layer
                            if (index === 0 && dataset.xs && dataset.xs[0]) {
                                layerConfig.inputShape = [dataset.xs[0].length];
                            }
                        }

                        return layerConfig;
                    }),
                    learningRate: learningRate,
                    loss: problemType === "classification" ?
                        (targetShape > 2 ? 'sparseCategoricalCrossentropy' : 'binaryCrossentropy') : 'meanSquaredError',
                    metrics: problemType === "classification" ? ['accuracy'] : ['mae']
                };

                setModelConfig(fixedModelConfig);

                console.log('Autofix applied: Model configuration updated with corrected shapes');
            }
        }

        // Clear the error and close dialog
        clearError();
        setShowErrorDialog(false);
    };

    // Generate initial sample data if no data exists
    useEffect(() => {
        console.log('Training page mounted, checking for data...');
        console.log('DataStore dataset:', !!dataStore.dataset);
        console.log('Training dataset:', !!dataset, 'size:', dataset?.xs?.length || 0);

        // If we have no data at all, load a preset to get started
        if (!dataStore.dataset && (!dataset || dataset.xs.length === 0)) {
            console.log('No data found, loading sinusoid preset...');
            dataStore.loadPreset('sinusoid');
        }
    }, []); // Empty dependency array to run only once on mount    // Transfer data when component mounts or when data store changes
    useEffect(() => {
        if (dataStore.dataset && dataStore.selectedFeatures.length > 0 && dataStore.selectedTarget) {
            const data = dataStore.dataset.data;
            const featureColumns = dataStore.selectedFeatures;
            const targetColumn = dataStore.selectedTarget;

            console.log('Transferring data to training:', {
                dataLength: data.length,
                featureColumns,
                targetColumn,
                sampleRow: data[0]
            });

            // Check if this is 2D classification data (x, y coordinates + class)
            const is2DClassification = featureColumns.length === 2 &&
                featureColumns.includes('x') && featureColumns.includes('y') &&
                targetColumn === 'class';

            let validData;
            if (is2DClassification) {
                // For 2D classification, use x,y coordinates and include class info
                validData = data
                    .map(row => ({
                        x: Number(row['x']),
                        y: Number(row['y']),
                        class: Number(row['class'])
                    }))
                    .filter(point => !isNaN(point.x) && !isNaN(point.y) && !isNaN(point.class));
            } else {
                // For 1D problems, use traditional mapping
                const featureColumn = featureColumns[0];
                validData = data
                    .map(row => ({
                        x: Number(row[featureColumn]),
                        y: Number(row[targetColumn])
                    }))
                    .filter(point => !isNaN(point.x) && !isNaN(point.y));
            }

            if (validData.length > 0) {
                const xs = validData.map(d => [d.x]);
                const ys = is2DClassification ? validData.map(d => [d.y]) : validData.map(d => [d.y]);

                console.log('Setting training dataset:', {
                    validDataLength: validData.length,
                    xsLength: xs.length,
                    ysLength: ys.length,
                    is2DClassification
                });

                setDataset(xs, ys);

                // Determine problem type
                if (is2DClassification) {
                    setProblemType("classification");
                } else {
                    const uniqueTargets = new Set(validData.map(d => d.y));
                    if (uniqueTargets.size <= 10 && Array.from(uniqueTargets).every(val => Number.isInteger(val))) {
                        setProblemType("classification");
                    } else {
                        setProblemType("regression");
                    }
                }
            }
        }
    }, [dataStore.dataset, dataStore.selectedFeatures, dataStore.selectedTarget, setDataset]);

    // Initialize worker on mount
    useEffect(() => {
        initializeWorker();

        return () => {
            disposeWorker();
        };
    }, [initializeWorker, disposeWorker]);

    // Sync hyperparameters to training configuration
    useEffect(() => {
        const actualBatchSize = batchSize === "Custom" ? customBatchSize || 32 : batchSize;

        setTrainingConfig({
            epochs,
            learningRate,
            optimizer: optimizer.toLowerCase(),
            batchSize: actualBatchSize,
        });

        console.log('Synced hyperparameters to training config:', {
            epochs,
            learningRate,
            optimizer: optimizer.toLowerCase(),
            batchSize: actualBatchSize,
        });
    }, [epochs, learningRate, optimizer, batchSize, customBatchSize, setTrainingConfig]);

    // Sync model layers to training configuration  
    useEffect(() => {
        if (layers && layers.length > 0 && dataset) {
            // Convert hyperparameters layers to worker format
            const modelLayers = layers.map((layer, index) => {
                const layerConfig: any = {
                    type: layer.type.toLowerCase(),
                };

                if (layer.type === "Dense") {
                    layerConfig.units = layer.units || 64;
                    layerConfig.activation = layer.activation?.toLowerCase() || 'relu';

                    // Add input shape for first layer
                    if (index === 0 && dataset.xs && dataset.xs[0]) {
                        layerConfig.inputShape = [dataset.xs[0].length];
                    }
                }

                return layerConfig;
            });

            const modelConfig = {
                layers: modelLayers,
                learningRate: learningRate,
                loss: problemType === "classification" ? 'sparseCategoricalCrossentropy' : 'meanSquaredError',
                metrics: problemType === "classification" ? ['accuracy'] : ['mae']
            };

            setModelConfig(modelConfig);

            console.log('Synced model layers to training config:', {
                layersCount: modelLayers.length,
                modelConfig,
                problemType
            });
        }
    }, [layers, learningRate, problemType, dataset, setModelConfig]);

    // Generate fallback sample data if no data exists
    useEffect(() => {
        // Only generate sample data if no data store data exists and no training dataset exists
        if (!dataStore.dataset && (!dataset || dataset.xs.length === 0)) {
            console.log('Generating sample data for problem type:', problemType);
            const sampleData = generateSampleData(problemType, 100);
            setDataset(
                sampleData.map(d => [d.x]),
                sampleData.map(d => [d.y])
            );
        }
    }, [problemType, dataStore.dataset, dataset?.xs?.length, setDataset]);

    const isTraining = trainingState === "training";
    const canStep = trainingState === "paused" || (trainingState === "idle" && speed === 0);

    const trainingData = useMemo(() => {
        if (!dataset || !dataset.xs || !dataset.ys) {
            console.log('No dataset available for training data');
            return [];
        }

        // Check if this is 2D classification data
        const is2DClassification = dataStore.selectedFeatures?.length === 2 &&
            dataStore.selectedFeatures.includes('x') && dataStore.selectedFeatures.includes('y') &&
            dataStore.selectedTarget === 'class';

        let data;
        if (is2DClassification && dataStore.dataset) {
            // For 2D classification, include class information
            data = dataset.xs.map((x, i) => {
                const originalRow = dataStore.dataset!.data[i];
                return {
                    x: x[0],
                    y: dataset.ys[i][0],
                    class: Number(originalRow['class'])
                };
            });
        } else {
            // Traditional mapping
            data = dataset.xs.map((x, i) => ({
                x: x[0],
                y: dataset.ys[i][0]
            }));
        }

        console.log('Training data prepared:', {
            datasetXsLength: dataset.xs.length,
            datasetYsLength: dataset.ys.length,
            trainingDataLength: data.length,
            sampleData: data.slice(0, 3),
            is2DClassification
        });

        return data;
    }, [dataset, dataStore.selectedFeatures, dataStore.selectedTarget, dataStore.dataset]);    // Debug log to check if we have training data
    console.log('Training data state:', {
        hasDataset: !!dataset,
        datasetXsLength: dataset?.xs?.length || 0,
        datasetYsLength: dataset?.ys?.length || 0,
        trainingDataLength: trainingData.length,
        sampleTrainingData: trainingData.slice(0, 3)
    });

    // Transform store predictions to match component expectations
    const transformedPredictions = modelPredictions.map(pred => ({
        x: pred.input,
        y: pred.prediction,
        prediction: pred.prediction,
        actual: pred.actual
    }));

    // Debug log to check if predictions are being received
    console.log('Model predictions count:', modelPredictions.length, 'Transformed:', transformedPredictions.length);

    return (
        <div className="p-6 space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight">Training</h1>
                    <p className="text-muted-foreground mt-2">
                        Train your neural network with real-time visualization and controls
                    </p>
                </div>
            </div>

            {!worker && (
                <Alert>
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>
                        Training worker is not available. Web Workers may not be supported in your browser.
                    </AlertDescription>
                </Alert>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Training Controls */}
                <div className="lg:col-span-1">
                    <TrainingControls
                        trainingState={trainingState}
                        speed={speed}
                        currentEpoch={currentEpoch}
                        totalEpochs={totalEpochs}
                        currentBatch={currentBatch}
                        totalBatches={totalBatches}
                        trainingMetrics={trainingMetrics}
                        onStart={startTraining}
                        onPause={pauseTraining}
                        onStop={stopTraining}
                        onSpeedChange={setSpeed}
                        onNextBatch={stepBatch}
                        onNextEpoch={stepEpoch}
                        onEpochScrub={scrubToEpoch}
                        canStep={canStep}
                    />
                </div>

                {/* Main Visualization */}
                <div className="lg:col-span-2">
                    <TrainingDataVisualization
                        data={trainingData}
                        modelPredictions={transformedPredictions}
                        isTraining={isTraining}
                        problemType={problemType}
                    />
                </div>
            </div>

            {/* Metrics and Analysis */}
            <Tabs defaultValue="metrics" className="w-full">
                <TabsList className="grid w-full grid-cols-3">
                    <TabsTrigger value="metrics" className="flex items-center gap-2">
                        <BarChart3 className="h-4 w-4" />
                        Training Metrics
                    </TabsTrigger>
                    <TabsTrigger value="activations" className="flex items-center gap-2">
                        <Activity className="h-4 w-4" />
                        Layer Activations
                    </TabsTrigger>
                    <TabsTrigger value="config" className="flex items-center gap-2">
                        <Brain className="h-4 w-4" />
                        Configuration
                    </TabsTrigger>
                </TabsList>

                <TabsContent value="metrics" className="mt-6">
                    <LiveMetricsChart
                        data={trainingMetrics}
                        isTraining={isTraining}
                        showValidation={trainingConfig.validationSplit > 0}
                    />
                </TabsContent>

                <TabsContent value="activations" className="mt-6">
                    <ActivationPanelGrid
                        layerData={layerActivations}
                        isTraining={isTraining}
                        maxPanels={6}
                    />
                </TabsContent>

                <TabsContent value="config" className="mt-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <Card>
                            <CardHeader>
                                <CardTitle>Training Configuration</CardTitle>
                                <CardDescription>
                                    Current training parameters
                                </CardDescription>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <div className="text-sm font-medium">Epochs</div>
                                        <div className="text-2xl font-bold">{trainingConfig.epochs}</div>
                                    </div>
                                    <div>
                                        <div className="text-sm font-medium">Batch Size</div>
                                        <div className="text-2xl font-bold">{trainingConfig.batchSize}</div>
                                    </div>
                                    <div>
                                        <div className="text-sm font-medium">Learning Rate</div>
                                        <div className="text-2xl font-bold">{trainingConfig.learningRate}</div>
                                    </div>
                                    <div>
                                        <div className="text-sm font-medium">Optimizer</div>
                                        <div className="text-2xl font-bold capitalize">{trainingConfig.optimizer}</div>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>

                        <Card>
                            <CardHeader>
                                <CardTitle>Dataset Information</CardTitle>
                                <CardDescription>
                                    Training data statistics
                                </CardDescription>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <div className="text-sm font-medium">Problem Type</div>
                                        <div className="text-2xl font-bold capitalize">{problemType}</div>
                                    </div>
                                    <div>
                                        <div className="text-sm font-medium">Samples</div>
                                        <div className="text-2xl font-bold">{trainingData.length}</div>
                                    </div>
                                    <div>
                                        <div className="text-sm font-medium">Validation Split</div>
                                        <div className="text-2xl font-bold">{(trainingConfig.validationSplit * 100).toFixed(0)}%</div>
                                    </div>
                                    <div>
                                        <div className="text-sm font-medium">Features</div>
                                        <div className="text-2xl font-bold">1</div>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                    </div>
                </TabsContent>
            </Tabs>

            {/* Error Dialog */}
            <Dialog open={showErrorDialog} onOpenChange={setShowErrorDialog}>
                <DialogContent className="sm:max-w-[500px]">
                    <DialogHeader>
                        <DialogTitle className="flex items-center gap-2">
                            <AlertCircle className="h-5 w-5 text-red-500" />
                            Training Error Detected
                        </DialogTitle>
                        <DialogDescription>
                            {errorType === 'shape_mismatch'
                                ? "A shape mismatch error occurred during training. This usually happens when the model's output layer doesn't match the target data shape."
                                : "An error occurred during training."
                            }
                        </DialogDescription>
                    </DialogHeader>

                    <div className="my-4">
                        <div className="text-sm font-medium mb-2">Error Details:</div>
                        <div className="p-3 bg-red-50 border border-red-200 rounded-md text-sm text-red-800 font-mono max-h-32 overflow-y-auto">
                            {lastError}
                        </div>
                    </div>

                    <DialogFooter className="flex gap-2">
                        <Button
                            variant="outline"
                            onClick={() => {
                                clearError();
                                setShowErrorDialog(false);
                            }}
                        >
                            Dismiss
                        </Button>
                        {errorType === 'shape_mismatch' && (
                            <Button
                                onClick={handleAutofix}
                                className="flex items-center gap-2"
                            >
                                <Wrench className="h-4 w-4" />
                                Auto-Fix Model
                            </Button>
                        )}
                    </DialogFooter>
                </DialogContent>
            </Dialog>
        </div>
    );
}