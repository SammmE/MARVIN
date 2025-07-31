import React from "react";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
} from "recharts";
import { useModelStore } from "../lib/model-store";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { cn } from "../lib/utils";

interface TrainingMetricsChartProps {
    className?: string;
}

export const TrainingMetricsChart: React.FC<TrainingMetricsChartProps> = ({
    className
}) => {
    const { trainingMetrics, currentMetrics } = useModelStore();

    // Format data for Recharts
    const chartData = trainingMetrics.map((metric, index) => ({
        epoch: metric.epoch,
        loss: metric.loss,
        accuracy: metric.accuracy ? metric.accuracy * 100 : null,
        valLoss: metric.valLoss,
        valAccuracy: metric.valAccuracy ? metric.valAccuracy * 100 : null,
        index,
    }));

    if (chartData.length === 0) {
        return (
            <Card className={cn("w-full", className)}>
                <CardHeader>
                    <CardTitle>Training Metrics</CardTitle>
                    <CardDescription>
                        Training metrics will appear here once training begins
                    </CardDescription>
                </CardHeader>
                <CardContent className="flex items-center justify-center h-64 text-muted-foreground">
                    No training data available
                </CardContent>
            </Card>
        );
    }

    return (
        <Card className={cn("w-full", className)}>
            <CardHeader>
                <CardTitle>Training Metrics</CardTitle>
                <CardDescription>
                    Real-time training progress and performance metrics
                </CardDescription>
                {currentMetrics && (
                    <div className="flex gap-4 text-sm">
                        <div>
                            <span className="font-medium">Current Loss:</span> {currentMetrics.loss.toFixed(6)}
                        </div>
                        {currentMetrics.accuracy && (
                            <div>
                                <span className="font-medium">Current Accuracy:</span> {(currentMetrics.accuracy * 100).toFixed(2)}%
                            </div>
                        )}
                    </div>
                )}
            </CardHeader>
            <CardContent>
                <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart
                            data={chartData}
                            margin={{
                                top: 5,
                                right: 30,
                                left: 20,
                                bottom: 5,
                            }}
                        >
                            <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                            <XAxis
                                dataKey="epoch"
                                className="text-muted-foreground"
                                label={{ value: "Epoch", position: "insideBottom", offset: -5 }}
                            />
                            <YAxis
                                yAxisId="loss"
                                orientation="left"
                                className="text-muted-foreground"
                                label={{ value: "Loss", angle: -90, position: "insideLeft" }}
                            />
                            <YAxis
                                yAxisId="accuracy"
                                orientation="right"
                                className="text-muted-foreground"
                                label={{ value: "Accuracy (%)", angle: 90, position: "insideRight" }}
                                domain={[0, 100]}
                            />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: "hsl(var(--card))",
                                    border: "1px solid hsl(var(--border))",
                                    borderRadius: "var(--radius)",
                                    color: "hsl(var(--card-foreground))",
                                }}
                                formatter={(value: number, name: string) => {
                                    if (name.includes("accuracy") || name.includes("Accuracy")) {
                                        return [`${value.toFixed(2)}%`, name];
                                    }
                                    return [value.toFixed(6), name];
                                }}
                                labelFormatter={(label) => `Epoch ${label}`}
                            />
                            <Legend />

                            {/* Loss Lines */}
                            <Line
                                yAxisId="loss"
                                type="monotone"
                                dataKey="loss"
                                stroke="hsl(var(--destructive))"
                                strokeWidth={2}
                                dot={{ fill: "hsl(var(--destructive))", strokeWidth: 2, r: 3 }}
                                activeDot={{ r: 5, stroke: "hsl(var(--destructive))", strokeWidth: 2 }}
                                name="Training Loss"
                            />
                            {chartData.some(d => d.valLoss !== undefined) && (
                                <Line
                                    yAxisId="loss"
                                    type="monotone"
                                    dataKey="valLoss"
                                    stroke="hsl(var(--destructive))"
                                    strokeWidth={2}
                                    strokeDasharray="5 5"
                                    dot={{ fill: "hsl(var(--destructive))", strokeWidth: 2, r: 3 }}
                                    activeDot={{ r: 5, stroke: "hsl(var(--destructive))", strokeWidth: 2 }}
                                    name="Validation Loss"
                                />
                            )}

                            {/* Accuracy Lines */}
                            {chartData.some(d => d.accuracy !== null) && (
                                <Line
                                    yAxisId="accuracy"
                                    type="monotone"
                                    dataKey="accuracy"
                                    stroke="hsl(var(--primary))"
                                    strokeWidth={2}
                                    dot={{ fill: "hsl(var(--primary))", strokeWidth: 2, r: 3 }}
                                    activeDot={{ r: 5, stroke: "hsl(var(--primary))", strokeWidth: 2 }}
                                    name="Training Accuracy"
                                />
                            )}
                            {chartData.some(d => d.valAccuracy !== null) && (
                                <Line
                                    yAxisId="accuracy"
                                    type="monotone"
                                    dataKey="valAccuracy"
                                    stroke="hsl(var(--primary))"
                                    strokeWidth={2}
                                    strokeDasharray="5 5"
                                    dot={{ fill: "hsl(var(--primary))", strokeWidth: 2, r: 3 }}
                                    activeDot={{ r: 5, stroke: "hsl(var(--primary))", strokeWidth: 2 }}
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
