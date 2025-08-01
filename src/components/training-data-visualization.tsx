import React, { useMemo } from "react";
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Line, ComposedChart, Legend } from "recharts";
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
        // For regression, create a dense set of prediction points to ensure smooth line
        if (problemType === "regression" && modelPredictions.length > 0) {
            // Get the range of input values
            const xValues = data.map(d => d.x).sort((a, b) => a - b);
            const minX = Math.min(...xValues);
            const maxX = Math.max(...xValues);

            // Create a denser set of points for smooth regression line
            const densePredictionPoints = [];
            const numPoints = 100; // Generate 100 points for smooth line

            for (let i = 0; i <= numPoints; i++) {
                const x = minX + (maxX - minX) * (i / numPoints);

                // Find the closest predictions to interpolate between
                const sortedPredictions = modelPredictions
                    .filter(pred => pred.x !== undefined && pred.y !== undefined)
                    .sort((a, b) => a.x - b.x);

                if (sortedPredictions.length > 0) {
                    let interpolatedY;

                    if (x <= sortedPredictions[0].x) {
                        // Use first prediction for values before the range
                        interpolatedY = sortedPredictions[0].y || sortedPredictions[0].prediction;
                    } else if (x >= sortedPredictions[sortedPredictions.length - 1].x) {
                        // Use last prediction for values after the range
                        interpolatedY = sortedPredictions[sortedPredictions.length - 1].y || sortedPredictions[sortedPredictions.length - 1].prediction;
                    } else {
                        // Interpolate between two nearest points
                        for (let j = 0; j < sortedPredictions.length - 1; j++) {
                            const pred1 = sortedPredictions[j];
                            const pred2 = sortedPredictions[j + 1];

                            if (x >= pred1.x && x <= pred2.x) {
                                const ratio = (x - pred1.x) / (pred2.x - pred1.x);
                                const y1 = pred1.y || pred1.prediction || 0;
                                const y2 = pred2.y || pred2.prediction || 0;
                                interpolatedY = y1 + ratio * (y2 - y1);
                                break;
                            }
                        }
                    }

                    if (interpolatedY !== undefined) {
                        densePredictionPoints.push({
                            x,
                            y: null, // No actual data at this point
                            prediction: interpolatedY,
                            residual: null
                        });
                    }
                }
            }

            // Combine actual data points with dense prediction points
            const actualDataPoints = data.map((point) => {
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

            // Combine and sort all points
            const allPoints = [...actualDataPoints, ...densePredictionPoints].sort((a, b) => a.x - b.x);

            console.log('Dense chartData prepared for regression:', {
                actualDataPoints: actualDataPoints.length,
                densePredictionPoints: densePredictionPoints.length,
                totalPoints: allPoints.length,
                samplePredictions: allPoints.filter(p => p.prediction !== null).slice(0, 5)
            });

            return allPoints;
        } else {
            // For classification or when no predictions, use original logic
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

            console.log('ChartData prepared for classification:', {
                dataLength: sortedData.length,
                sampledData: sortedData.slice(0, 3),
                predictionValues: sortedData.map(d => d.prediction).filter(p => p !== null).slice(0, 5)
            });

            return sortedData;
        }
    }, [data, modelPredictions, problemType]);

    // Determine if we have classification data with distinct classes
    const classGroups = useMemo(() => {
        if (problemType === "classification") {
            const uniqueClasses = Array.from(new Set(data.map(d => d.y)));
            // Only treat as classes if we have <= 10 unique values AND they appear to be categorical
            if (uniqueClasses.length <= 10 && uniqueClasses.every(val =>
                typeof val === 'string' || Number.isInteger(val)
            )) {
                return uniqueClasses.sort();
            }
        }
        return null;
    }, [data, problemType]);

    // Colors for different classes
    const COLORS = [
        "#3b82f6", // blue
        "#ef4444", // red  
        "#10b981", // green
        "#f59e0b", // yellow
        "#8b5cf6", // purple
        "#06b6d4", // cyan
        "#84cc16", // lime
        "#f97316", // orange
        "#ec4899", // pink
        "#6b7280", // gray
    ];

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
                                    data={chartData.filter(d => d.y !== null)}
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
                                        connectNulls={true}
                                        isAnimationActive={false}
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
                                    name="Output"
                                    domain={['dataMin - 0.1', 'dataMax + 0.1']}
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

                                {/* Legend for classes */}
                                {classGroups && <Legend />}

                                {/* Actual data points colored by class */}
                                {classGroups ? (
                                    // Show different colors for each class
                                    classGroups.map((classValue, index) => (
                                        <Scatter
                                            key={`class-${classValue}`}
                                            name={`Class ${classValue}`}
                                            data={chartData.filter(d => d.y === classValue)}
                                            fill={COLORS[index % COLORS.length]}
                                            dataKey="y"
                                        />
                                    ))
                                ) : (
                                    // Single color for all data points when not classification
                                    <Scatter
                                        dataKey="y"
                                        fill="#3b82f6"
                                        name="Training Data"
                                    />
                                )}

                                {/* Model predictions as crosses */}
                                {modelPredictions.length > 0 && (
                                    classGroups ? (
                                        // Show different colors for each predicted class
                                        classGroups.map((classValue, index) => {
                                            const predictionColor = COLORS[(index + 4) % COLORS.length]; // Offset colors to distinguish from training data
                                            return (
                                                <Scatter
                                                    key={`prediction-class-${classValue}`}
                                                    name={`Predicted Class ${classValue}`}
                                                    data={chartData.filter(d => d.prediction === classValue)}
                                                    fill={predictionColor}
                                                    shape="cross"
                                                    dataKey="prediction"
                                                />
                                            );
                                        })
                                    ) : (
                                        // Single color for all predictions when not classification
                                        <Scatter
                                            dataKey="prediction"
                                            fill="#ef4444"
                                            shape="cross"
                                            name="Model Predictions"
                                        />
                                    )
                                )}
                            </ScatterChart>
                        )}
                    </ResponsiveContainer>
                </div>

                {/* Legend and stats */}
                <div className="mt-4 flex items-center justify-between text-sm">
                    <div className="flex items-center gap-4">
                        {classGroups ? (
                            // Show class breakdown when we have classes
                            classGroups.map((classValue, index) => {
                                const classCount = data.filter(d => d.y === classValue).length;
                                return (
                                    <div key={`legend-${classValue}`} className="flex items-center gap-2">
                                        <div
                                            className="w-3 h-3 rounded-full"
                                            style={{ backgroundColor: COLORS[index % COLORS.length] }}
                                        ></div>
                                        <span>Class {classValue} ({classCount} points)</span>
                                    </div>
                                );
                            })
                        ) : (
                            // Single entry for non-classification data
                            <div className="flex items-center gap-2">
                                <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                                <span>Training Data ({data.length} points)</span>
                            </div>
                        )}

                        {modelPredictions.length > 0 && (
                            classGroups ? (
                                // Show prediction legend for each class
                                classGroups.map((classValue, index) => {
                                    const predictionColor = COLORS[(index + 4) % COLORS.length];
                                    const predictionCount = chartData.filter(d => d.prediction === classValue).length;
                                    return predictionCount > 0 ? (
                                        <div key={`pred-legend-${classValue}`} className="flex items-center gap-2">
                                            <div className="w-3 h-3 flex items-center justify-center text-white text-xs font-bold" style={{ backgroundColor: predictionColor }}>
                                                ×
                                            </div>
                                            <span>Predicted Class {classValue} ({predictionCount})</span>
                                        </div>
                                    ) : null;
                                })
                            ) : (
                                // Single prediction legend for non-classification
                                <div className="flex items-center gap-2">
                                    <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                                    <span>
                                        {problemType === "regression" ? "Regression Line" : "Predictions"}
                                    </span>
                                </div>
                            )
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
