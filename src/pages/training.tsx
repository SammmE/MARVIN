import React, { useEffect, useState, useMemo } from "react";
import { TrainingControls } from "../components/training-controls";
import { LiveMetricsChart } from "../components/live-metrics-chart";
import { ActivationPanelGrid } from "../components/activation-mini-panels";
import { TrainingDataVisualization } from "../components/training-data-visualization";
import { useTrainingStore } from "../lib/training-store";
import { useDataStore } from "../lib/data-store";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";
import { Alert, AlertDescription } from "../components/ui/alert";
import { AlertCircle, Brain, BarChart3, Activity } from "lucide-react";

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
        initializeWorker,
        disposeWorker,
    } = useTrainingStore();

    // Get data from the data store
    const dataStore = useDataStore();

    const [problemType, setProblemType] = useState<"regression" | "classification">("regression");

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
            const featureColumn = dataStore.selectedFeatures[0];
            const targetColumn = dataStore.selectedTarget;

            console.log('Transferring data to training:', {
                dataLength: data.length,
                featureColumn,
                targetColumn,
                sampleRow: data[0]
            });

            // Convert to training format
            const validData = data
                .map(row => ({
                    x: Number(row[featureColumn]),
                    y: Number(row[targetColumn])
                }))
                .filter(point => !isNaN(point.x) && !isNaN(point.y));

            if (validData.length > 0) {
                const xs = validData.map(d => [d.x]);
                const ys = validData.map(d => [d.y]);

                console.log('Setting training dataset:', {
                    validDataLength: validData.length,
                    xsLength: xs.length,
                    ysLength: ys.length
                });

                setDataset(xs, ys);

                // Determine problem type
                const uniqueTargets = new Set(validData.map(d => d.y));
                if (uniqueTargets.size <= 10 && Array.from(uniqueTargets).every(val => Number.isInteger(val))) {
                    setProblemType("classification");
                } else {
                    setProblemType("regression");
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

        const data = dataset.xs.map((x, i) => ({
            x: x[0],
            y: dataset.ys[i][0]
        }));

        console.log('Training data prepared:', {
            datasetXsLength: dataset.xs.length,
            datasetYsLength: dataset.ys.length,
            trainingDataLength: data.length,
            sampleData: data.slice(0, 3)
        });

        return data;
    }, [dataset]);

    // Debug log to check if we have training data
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
        </div>
    );
}