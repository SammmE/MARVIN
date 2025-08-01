import React from "react";
import { BarChart, Bar, ResponsiveContainer, Cell } from "recharts";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Activity } from "lucide-react";
import type { LayerActivationData } from "../lib/oscar-store";

interface ActivationMiniPanelProps {
    data: LayerActivationData;
    showGradients?: boolean;
    showWeights?: boolean;
    isLive?: boolean;
}

export const ActivationMiniPanel: React.FC<ActivationMiniPanelProps> = ({
    data,
    showGradients = true,
    showWeights = false,
    isLive = false,
}) => {
    const { layerName, activations, gradients, weights } = data;

    // Prepare activation data for bar chart
    const activationData = activations.slice(0, 20).map((value, index) => ({
        neuron: index,
        activation: value,
        gradient: gradients?.[index] || 0,
    }));

    // Calculate statistics
    const activationStats = {
        mean: activations.reduce((sum, val) => sum + val, 0) / activations.length,
        max: Math.max(...activations),
        min: Math.min(...activations),
        nonZero: activations.filter(val => Math.abs(val) > 0.001).length,
    };

    const weightStats = weights ? {
        mean: weights.flat().reduce((sum, val) => sum + val, 0) / weights.flat().length,
        max: Math.max(...weights.flat()),
        min: Math.min(...weights.flat()),
        count: weights.flat().length,
    } : null;

    return (
        <Card className="w-full">
            <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                    <CardTitle className="text-sm flex items-center gap-2">
                        <Activity className="h-4 w-4" />
                        {layerName}
                    </CardTitle>
                    {isLive && (
                        <Badge variant="default" className="animate-pulse text-xs">
                            Live
                        </Badge>
                    )}
                </div>
                <CardDescription className="text-xs">
                    {activations.length} neurons • {activationStats.nonZero} active
                </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
                {/* Activation Bars */}
                <div>
                    <div className="text-xs font-medium mb-2">Activations</div>
                    <div className="h-16">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={activationData} margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
                                <Bar dataKey="activation" radius={1}>
                                    {activationData.map((entry, index) => (
                                        <Cell
                                            key={`cell-${index}`}
                                            fill={entry.activation > 0 ? "#22c55e" : entry.activation < 0 ? "#ef4444" : "#64748b"}
                                        />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Gradient Bars */}
                {showGradients && gradients && (
                    <div>
                        <div className="text-xs font-medium mb-2">Gradients</div>
                        <div className="h-12">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={activationData} margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
                                    <Bar dataKey="gradient" radius={1}>
                                        {activationData.map((entry, index) => (
                                            <Cell
                                                key={`grad-${index}`}
                                                fill={entry.gradient > 0 ? "#3b82f6" : entry.gradient < 0 ? "#f59e0b" : "#64748b"}
                                            />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                )}

                {/* Statistics */}
                <div className="space-y-2">
                    <div className="flex justify-between text-xs">
                        <span className="text-muted-foreground">Activation Stats:</span>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                        <div className="flex justify-between">
                            <span>Mean:</span>
                            <span className="font-mono">{activationStats.mean.toFixed(3)}</span>
                        </div>
                        <div className="flex justify-between">
                            <span>Range:</span>
                            <span className="font-mono">
                                {activationStats.min.toFixed(2)} to {activationStats.max.toFixed(2)}
                            </span>
                        </div>
                    </div>

                    {weightStats && showWeights && (
                        <>
                            <div className="flex justify-between text-xs pt-2 border-t">
                                <span className="text-muted-foreground">Weight Stats:</span>
                            </div>
                            <div className="grid grid-cols-2 gap-2 text-xs">
                                <div className="flex justify-between">
                                    <span>W Mean:</span>
                                    <span className="font-mono">{weightStats.mean.toFixed(3)}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span>W Count:</span>
                                    <span className="font-mono">{weightStats.count}</span>
                                </div>
                            </div>
                        </>
                    )}
                </div>
            </CardContent>
        </Card>
    );
};

interface ActivationPanelGridProps {
    layerData: LayerActivationData[];
    isTraining: boolean;
    maxPanels?: number;
}

export const ActivationPanelGrid: React.FC<ActivationPanelGridProps> = ({
    layerData,
    isTraining,
    maxPanels = 6,
}) => {
    const displayLayers = layerData.slice(0, maxPanels);

    return (
        <div className="space-y-4">
            <div className="flex items-center justify-between">
                <h3 className="text-lg font-medium">Layer Activations</h3>
                {isTraining && (
                    <Badge variant="default" className="animate-pulse">
                        Live Updates
                    </Badge>
                )}
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {displayLayers.map((layer) => (
                    <ActivationMiniPanel
                        key={layer.layerId}
                        data={layer}
                        isLive={isTraining}
                        showGradients={isTraining}
                        showWeights={false}
                    />
                ))}
            </div>
            {layerData.length > maxPanels && (
                <div className="text-center text-sm text-muted-foreground">
                    Showing {maxPanels} of {layerData.length} layers
                </div>
            )}
        </div>
    );
};
