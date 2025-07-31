import { useDataStore } from "@/lib/data-store";
import { Card, CardHeader, CardTitle, CardContent } from "./ui/card";
import { ChartContainer } from "./ui/chart";
import {
    LineChart,
    Line,
    ScatterChart,
    Scatter,
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Brush,
    Legend,
} from "recharts";
import { Button } from "./ui/button";
import { useState } from "react";

const COLORS = [
    "#8884d8",
    "#82ca9d",
    "#ffc658",
    "#ff8042",
    "#a4de6c",
    "#d0ed57",
    "#83a6ed",
    "#8dd1e1",
    "#82b1ff",
    "#c792ea",
];

export function DataVisualization() {
    const {
        dataset,
        chartType,
        xAxis,
        yAxis,
        brushEnabled,
        selectedTarget,
        toggleBrush,
    } = useDataStore();

    const [zoomDomain, setZoomDomain] = useState<{
        x?: [number, number];
        y?: [number, number];
    } | null>(null);

    if (!dataset || !xAxis || !yAxis) {
        return (
            <Card>
                <CardHeader>
                    <CardTitle>Visualization</CardTitle>
                </CardHeader>
                <CardContent className="text-muted-foreground text-center min-h-[300px] flex items-center justify-center">
                    <p>Select data and axes to visualize</p>
                </CardContent>
            </Card>
        );
    }

    const handleResetZoom = () => {
        setZoomDomain(null);
    };

    // Common chart configuration
    const chartConfig = {
        margin: { top: 20, right: 30, left: 20, bottom: 60 },
    };

    // Prepare data groups if we have a target variable AND it's categorical (classification)
    const targetGroups = selectedTarget ? (() => {
        const uniqueTargets = Array.from(new Set(dataset.data.map((d) => d[selectedTarget])));
        // Only treat as classes if we have <= 10 unique values AND they appear to be categorical
        if (uniqueTargets.length <= 10 && uniqueTargets.every(val =>
            typeof val === 'string' || Number.isInteger(val)
        )) {
            return uniqueTargets;
        }
        return null; // Treat as regression/continuous data
    })() : null;

    // Render the appropriate chart based on selected type
    const renderChart = () => {
        if (chartType === "scatter") {
            return (
                <ScatterChart data={dataset.data} {...chartConfig}>
                    <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                    <XAxis
                        type="number"
                        dataKey={xAxis}
                        name="Input"
                        label={{ value: xAxis, position: "insideBottom", offset: -5 }}
                        domain={zoomDomain?.x ? zoomDomain.x : ['dataMin - 0.1', 'dataMax + 0.1']}
                    />
                    <YAxis
                        type="number"
                        dataKey={yAxis}
                        name="Output"
                        label={{
                            value: yAxis,
                            angle: -90,
                            position: "insideLeft",
                            offset: 10,
                        }}
                        domain={zoomDomain?.y ? zoomDomain.y : ['dataMin - 0.1', 'dataMax + 0.1']}
                    />
                    <Tooltip
                        cursor={{ strokeDasharray: "3 3" }}
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

                    {/* Show legend only when we have classes to distinguish */}
                    {targetGroups && <Legend />}

                    {/* Show data points with class distinction if target exists, otherwise single color */}
                    {targetGroups ? (
                        // If we have a target variable (classes), group by it with different colors
                        targetGroups.map((group, index) => (
                            <Scatter
                                key={`scatter-${group}`}
                                name={`${selectedTarget}: ${group}`}
                                data={dataset.data.filter(
                                    (d) => d[selectedTarget as string] === group,
                                )}
                                fill={COLORS[index % COLORS.length]}
                                dataKey={yAxis}
                                shape="circle"
                            />
                        ))
                    ) : (
                        // Single scatter plot for all data points when no classes
                        <Scatter
                            dataKey={yAxis}
                            fill="#3b82f6"
                            name="Training Data"
                        />
                    )}

                    {brushEnabled && (
                        <Brush
                            dataKey={xAxis}
                            height={30}
                            stroke="#8884d8"
                            y={250}
                            startIndex={0}
                            endIndex={Math.min(50, dataset.data.length - 1)}
                        />
                    )}
                </ScatterChart>
            );
        } else if (chartType === "line") {
            return (
                <LineChart data={dataset.data} {...chartConfig}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                        dataKey={xAxis}
                        name={xAxis}
                        label={{ value: xAxis, position: "insideBottom", offset: -5 }}
                        domain={zoomDomain?.x ? zoomDomain.x : ["auto", "auto"]}
                    />
                    <YAxis
                        dataKey={yAxis}
                        name={yAxis}
                        label={{
                            value: yAxis,
                            angle: -90,
                            position: "insideLeft",
                            offset: 10,
                        }}
                        domain={zoomDomain?.y ? zoomDomain.y : ["auto", "auto"]}
                    />
                    <Tooltip
                        content={({ active, payload }) => {
                            if (active && payload && payload.length) {
                                return (
                                    <div className="bg-white p-2 border border-gray-200 rounded shadow-lg">
                                        <p className="text-sm font-medium">Data Point</p>
                                    </div>
                                );
                            }
                            return null;
                        }}
                    />
                    <Legend />

                    {targetGroups ? (
                        // If we have a target variable, group by it
                        targetGroups.map((group, index) => (
                            <Line
                                key={`line-${group}`}
                                type="monotone"
                                name={`${selectedTarget}: ${group}`}
                                data={dataset.data.filter(
                                    (d) => d[selectedTarget as string] === group,
                                )}
                                dataKey={yAxis}
                                stroke={COLORS[index % COLORS.length]}
                                dot={{ r: 3 }}
                                isAnimationActive={false}
                            />
                        ))
                    ) : (
                        // Otherwise just plot a single line
                        <Line
                            type="monotone"
                            dataKey={yAxis}
                            stroke={COLORS[0]}
                            dot={{ r: 3 }}
                            isAnimationActive={false}
                        />
                    )}

                    {brushEnabled && (
                        <Brush
                            dataKey={xAxis}
                            height={30}
                            stroke="#8884d8"
                            y={250}
                            startIndex={0}
                            endIndex={Math.min(50, dataset.data.length - 1)}
                        />
                    )}
                </LineChart>
            );
        } else if (chartType === "histogram") {
            // For histogram we'll need to calculate bins
            const histogramData = processHistogramData(dataset.data, xAxis);

            return (
                <BarChart data={histogramData} {...chartConfig}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                        dataKey="binLabel"
                        label={{ value: xAxis, position: "insideBottom", offset: -5 }}
                    />
                    <YAxis
                        label={{
                            value: "Count",
                            angle: -90,
                            position: "insideLeft",
                            offset: 10,
                        }}
                    />
                    <Tooltip
                        content={({ active, payload }) => {
                            if (active && payload && payload.length) {
                                return (
                                    <div className="bg-white p-2 border border-gray-200 rounded shadow-lg">
                                        <p className="text-sm font-medium">Histogram Bin</p>
                                    </div>
                                );
                            }
                            return null;
                        }}
                    />
                    <Legend />
                    <Bar dataKey="count" fill={COLORS[0]} name="Count" />
                </BarChart>
            );
        } else {
            return <div>Unsupported chart type</div>;
        }
    };

    return (
        <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle>Visualization</CardTitle>
                <div className="space-x-2">
                    <Button variant="outline" size="sm" onClick={toggleBrush}>
                        {brushEnabled ? "Disable Brush" : "Enable Brush"}
                    </Button>
                    {zoomDomain && (
                        <Button variant="outline" size="sm" onClick={handleResetZoom}>
                            Reset Zoom
                        </Button>
                    )}
                </div>
            </CardHeader>
            <CardContent className="p-0 pt-2 h-[400px]">
                <ChartContainer
                    config={{ primary: { color: "#8884d8" } }}
                    className="h-full"
                >
                    {renderChart()}
                </ChartContainer>
            </CardContent>
        </Card>
    );
}

// Helper function to process data for histogram
function processHistogramData(
    data: Record<string, unknown>[],
    key: string,
): { binLabel: string; count: number }[] {
    // For numeric data, calculate bins
    const values = data.map((d) => Number(d[key])).filter((v) => !isNaN(v));

    if (values.length === 0) return [];

    // Simple binning logic - 10 bins
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min;
    const binSize = range / 10;

    const bins: { binLabel: string; count: number }[] = [];

    for (let i = 0; i < 10; i++) {
        const binStart = min + i * binSize;
        const binEnd = binStart + binSize;
        const count = values.filter((v) => v >= binStart && v < binEnd).length;
        bins.push({
            binLabel: `${binStart.toFixed(2)} - ${binEnd.toFixed(2)}`,
            count,
        });
    }

    return bins;
}
