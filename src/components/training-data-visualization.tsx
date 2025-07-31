import React, { useMemo } from "react";
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Line, ReferenceLine, ComposedChart } from "recharts";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Activity, TrendingUp } from "lucide-react";

interface DataPoint {
    x: number;
    y: number;
    prediction?: number;
    actual?: number;
}

interface TrainingDataVisualizationProps {
    data: DataPoint[];
    modelPredictions?: DataPoint[];
    isTraining: boolean;
    problemType: "regression" | "classification";
}

export const TrainingDataVisualization: React.FC<TrainingDataVisualizationProps> = ({
    data,
    modelPredictions = [],
    isTraining,
    problemType,
}) => {
    // Debug logging
    console.log('TrainingDataVisualization received:', {
        dataLength: data.length,
        modelPredictionsLength: modelPredictions.length,
        modelPredictions: modelPredictions.slice(0, 3), // Show first 3 for debugging
        problemType,
        isTraining,
        sampleData: data.slice(0, 3) // Show first 3 data points
    });

    // Prepare data for visualization
    const chartData = useMemo(() => {
        const mappedData = data.map((point) => {
            // Find matching prediction by input value (more reliable than index)
            const matchingPrediction = modelPredictions.find(pred =>
                Math.abs(pred.x - point.x) < 0.001
            );

            return {
                ...point,
                prediction: matchingPrediction?.y || matchingPrediction?.prediction || null,
                residual: matchingPrediction ?
                    Math.abs(point.y - (matchingPrediction.y || matchingPrediction.prediction || 0)) : null
            };
        });

        // Sort by x value for proper line connection
        const sortedData = mappedData.sort((a, b) => a.x - b.x);

        console.log('ChartData prepared:', {
            dataLength: sortedData.length,
            sampledData: sortedData.slice(0, 3),
            predictionValues: sortedData.map(d => d.prediction).filter(p => p !== null).slice(0, 5)
        });

        return sortedData;
    }, [data, modelPredictions]);

    return (
        <Card className="w-full">
            <CardHeader>
                <div className="flex items-center justify-between">
                    <div>
                        <CardTitle className="flex items-center gap-2">
                            <Activity className="h-5 w-5" />
                            Training Data & Model Output
                            {isTraining && (
                                <Badge variant="default" className="animate-pulse">
                                    Live
                                </Badge>
                            )}
                        </CardTitle>
                        <CardDescription>
                            {problemType === "regression" ? "Regression line" : "Decision boundary"} overlaid on training data
                        </CardDescription>
                    </div>

                    <Badge variant="outline">
                        {problemType.charAt(0).toUpperCase() + problemType.slice(1)}
                    </Badge>
                </div>
            </CardHeader>
            <CardContent>
                <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                        {problemType === "regression" ? (
                            <ComposedChart data={chartData}>
                                <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                                <XAxis
                                    type="number"
                                    dataKey="x"
                                    name="Input"
                                    domain={['dataMin - 0.1', 'dataMax + 0.1']}
                                />
                                <YAxis
                                    type="number"
                                    dataKey="y"
                                    name="Output"
                                    domain={['dataMin - 0.1', 'dataMax + 0.1']}
                                />
                                <Tooltip
                                    content={({ active, payload }) => {
                                        if (active && payload && payload.length) {
                                            return (
                                                <div className="bg-white p-2 border border-gray-200 rounded shadow-lg">
                                                    <p className="text-sm font-medium">Training Data Point</p>
                                                </div>
                                            );
                                        }
                                        return null;
                                    }}
                                />

                                {/* Actual data points */}
                                <Scatter
                                    dataKey="y"
                                    fill="#3b82f6"
                                    name="Training Data"
                                />

                                {/* Model predictions as line if available */}
                                {modelPredictions.length > 0 && (
                                    <Line
                                        type="monotone"
                                        dataKey="prediction"
                                        stroke="#ef4444"
                                        strokeWidth={2}
                                        dot={false}
                                        name="Model Prediction"
                                        connectNulls={false}
                                    />
                                )}
                            </ComposedChart>
                        ) : (
                            <ScatterChart data={chartData}>
                                <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                                <XAxis
                                    type="number"
                                    dataKey="x"
                                    name="Feature"
                                    domain={['dataMin - 0.1', 'dataMax + 0.1']}
                                />
                                <YAxis
                                    type="number"
                                    dataKey="y"
                                    name="Class"
                                    domain={[-0.1, 1.1]}
                                    tickCount={3}
                                    tickFormatter={(value) => value === 0 ? 'Class 0' : value === 1 ? 'Class 1' : ''}
                                />
                                <Tooltip
                                    content={({ active, payload }) => {
                                        if (active && payload && payload.length) {
                                            return (
                                                <div className="bg-white p-2 border border-gray-200 rounded shadow-lg">
                                                    <p className="text-sm font-medium">Training Data</p>
                                                </div>
                                            );
                                        }
                                        return null;
                                    }}
                                />

                                {/* Decision boundary at 0.5 */}
                                <ReferenceLine y={0.5} stroke="#64748b" strokeDasharray="5 5" />

                                {/* Actual data points colored by class */}
                                <Scatter
                                    dataKey="y"
                                    fill="#3b82f6"
                                    name="Training Data"
                                />

                                {/* Model predictions as crosses */}
                                {modelPredictions.length > 0 && (
                                    <Scatter
                                        dataKey="prediction"
                                        fill="#ef4444"
                                        shape="cross"
                                        name="Model Predictions"
                                    />
                                )}
                            </ScatterChart>
                        )}
                    </ResponsiveContainer>
                </div>

                {/* Legend and stats */}
                <div className="mt-4 flex items-center justify-between text-sm">
                    <div className="flex items-center gap-4">
                        <div className="flex items-center gap-2">
                            <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                            <span>Training Data ({data.length} points)</span>
                        </div>
                        {modelPredictions.length > 0 && (
                            <div className="flex items-center gap-2">
                                <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                                <span>
                                    {problemType === "regression" ? "Regression Line" : "Predictions"}
                                </span>
                            </div>
                        )}
                    </div>

                    {isTraining && (
                        <Badge variant="outline" className="flex items-center gap-1">
                            <TrendingUp className="h-3 w-3" />
                            Updating Live
                        </Badge>
                    )}
                </div>
            </CardContent>
        </Card>
    );
};
