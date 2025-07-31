import React, { useMemo } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { TrendingUp, TrendingDown } from "lucide-react";

export interface MetricData {
    epoch: number;
    batch?: number;
    loss: number;
    accuracy?: number;
    valLoss?: number;
    valAccuracy?: number;
    timestamp: number;
}

interface LiveMetricsChartProps {
    data: MetricData[];
    isTraining: boolean;
    showValidation?: boolean;
}

export const LiveMetricsChart: React.FC<LiveMetricsChartProps> = ({
    data,
    isTraining,
    showValidation = true,
}) => {
    const latestMetrics = data[data.length - 1];

    const chartData = useMemo(() => {
        return data.map((point, index) => ({
            ...point,
            step: index,
        }));
    }, [data]);

    const getLossStatus = () => {
        if (data.length < 2) return null;
        const current = data[data.length - 1]?.loss || 0;
        const previous = data[data.length - 2]?.loss || 0;
        return current < previous ? "improving" : "worsening";
    };

    const lossStatus = getLossStatus();

    return (
        <Card className="w-full">
            <CardHeader>
                <div className="flex items-center justify-between">
                    <div>
                        <CardTitle className="flex items-center gap-2">
                            Training Metrics
                            {isTraining && (
                                <Badge variant="default" className="animate-pulse">
                                    Live
                                </Badge>
                            )}
                        </CardTitle>
                        <CardDescription>
                            Real-time loss and accuracy during training
                        </CardDescription>
                    </div>
                    {latestMetrics && (
                        <div className="text-right">
                            <div className="flex items-center gap-1 text-sm">
                                {lossStatus === "improving" ? (
                                    <TrendingDown className="h-4 w-4 text-green-500" />
                                ) : lossStatus === "worsening" ? (
                                    <TrendingUp className="h-4 w-4 text-red-500" />
                                ) : null}
                                <span className="font-medium">
                                    Loss: {latestMetrics.loss.toFixed(4)}
                                </span>
                            </div>
                            {latestMetrics.accuracy !== undefined && (
                                <div className="text-sm text-muted-foreground">
                                    Accuracy: {(latestMetrics.accuracy * 100).toFixed(2)}%
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </CardHeader>
            <CardContent>
                <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartData}>
                            <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                            <XAxis
                                dataKey="step"
                                label={{ value: 'Training Step', position: 'insideBottom', offset: -5 }}
                            />
                            <YAxis
                                yAxisId="loss"
                                orientation="left"
                                label={{ value: 'Loss', angle: -90, position: 'insideLeft' }}
                            />
                            {latestMetrics?.accuracy !== undefined && (
                                <YAxis
                                    yAxisId="accuracy"
                                    orientation="right"
                                    domain={[0, 1]}
                                    label={{ value: 'Accuracy', angle: 90, position: 'insideRight' }}
                                />
                            )}
                            <Tooltip
                                formatter={(value, name) => [
                                    typeof value === 'number' ? value.toFixed(4) : value,
                                    name
                                ]}
                                labelFormatter={(step) => `Step ${step}`}
                            />
                            <Legend />

                            <Line
                                yAxisId="loss"
                                type="monotone"
                                dataKey="loss"
                                stroke="#ef4444"
                                strokeWidth={2}
                                dot={false}
                                name="Training Loss"
                            />

                            {showValidation && (
                                <Line
                                    yAxisId="loss"
                                    type="monotone"
                                    dataKey="valLoss"
                                    stroke="#f97316"
                                    strokeWidth={2}
                                    strokeDasharray="5 5"
                                    dot={false}
                                    name="Validation Loss"
                                />
                            )}

                            {latestMetrics?.accuracy !== undefined && (
                                <Line
                                    yAxisId="accuracy"
                                    type="monotone"
                                    dataKey="accuracy"
                                    stroke="#22c55e"
                                    strokeWidth={2}
                                    dot={false}
                                    name="Training Accuracy"
                                />
                            )}

                            {showValidation && latestMetrics?.valAccuracy !== undefined && (
                                <Line
                                    yAxisId="accuracy"
                                    type="monotone"
                                    dataKey="valAccuracy"
                                    stroke="#16a34a"
                                    strokeWidth={2}
                                    strokeDasharray="5 5"
                                    dot={false}
                                    name="Validation Accuracy"
                                />
                            )}
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </CardContent>
        </Card>
    );
};
