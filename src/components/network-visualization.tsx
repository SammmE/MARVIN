import React, { useCallback, useMemo } from "react";
import {
    ReactFlow,
    useNodesState,
    useEdgesState,
    addEdge,
    Background,
    Controls,
    MiniMap,
    Handle,
    Position,
    Panel,
} from "@xyflow/react";
import type {
    Node,
    Edge,
    Connection,
    NodeTypes,
    EdgeTypes,
} from "@xyflow/react";
import { useModelStore, type LayerNode as LayerNodeType, type NeuronNode as NeuronNodeType } from "../lib/oscar-store";
import { cn } from "../lib/utils";
import { Button } from "./ui/button";
import { Minimize, Eye, EyeOff } from "lucide-react";

// Type definitions for node data
interface LayerNodeData {
    layer: LayerNodeType;
    stats?: {
        mean: number;
        max: number;
        min: number;
        std: number;
    };
}

interface NeuronNodeData {
    neuron: NeuronNodeType;
}

interface WeightEdgeData {
    weight: number;
    thickness: number;
}

interface WeightEdgeProps {
    id: string;
    sourceX: number;
    sourceY: number;
    targetX: number;
    targetY: number;
    data: WeightEdgeData;
}

// Custom Layer Node Component
const LayerNode: React.FC<{ data: LayerNodeData }> = ({ data }) => {
    const { layer, stats } = data;
    const { toggleLayerExpansion } = useModelStore();

    // Calculate color intensity based on activation magnitude
    const intensity = Math.min(1, Math.abs(layer.activationMagnitude) / 10);
    const hue = layer.activationMagnitude >= 0 ? 120 : 0; // Green for positive, red for negative

    // Different colors for input/output layers
    let borderColor = `hsla(${hue}, 70%, 40%, ${intensity * 0.5 + 0.3})`;
    let backgroundColor = `hsla(${hue}, 70%, 50%, ${intensity * 0.3 + 0.1})`;

    if (layer.isInput) {
        borderColor = 'hsl(220, 70%, 50%)'; // Blue for input
        backgroundColor = 'hsl(220, 70%, 95%)';
    } else if (layer.isOutput) {
        borderColor = 'hsl(280, 70%, 50%)'; // Purple for output  
        backgroundColor = 'hsl(280, 70%, 95%)';
    }

    const handleExpand = (e: React.MouseEvent) => {
        e.stopPropagation();
        toggleLayerExpansion(layer.id);
    };

    return (
        <div
            className={cn(
                "px-4 py-3 shadow-lg rounded-lg border-2",
                "bg-card text-card-foreground min-w-[120px] text-center relative",
                "hover:shadow-xl transition-all duration-200",
                layer.isExpanded && "ring-2 ring-blue-400"
            )}
            style={{
                backgroundColor,
                borderColor,
            }}
        >
            {/* Input handle - left side */}
            <Handle
                type="target"
                position={Position.Left}
                id="input"
                style={{ background: '#555', width: 8, height: 8 }}
            />

            {/* Layer type badge */}
            {(layer.isInput || layer.isOutput) && (
                <div className={cn(
                    "absolute -top-2 -right-2 px-2 py-1 text-xs rounded-full",
                    "text-white font-semibold",
                    layer.isInput ? "bg-blue-500" : "bg-purple-500"
                )}>
                    {layer.isInput ? "INPUT" : "OUTPUT"}
                </div>
            )}

            {/* Expand/Collapse button */}
            <button
                onClick={handleExpand}
                className={cn(
                    "absolute -top-2 -left-2 w-6 h-6 rounded-full",
                    "bg-gray-600 text-white text-xs hover:bg-gray-700",
                    "flex items-center justify-center transition-colors"
                )}
            >
                {layer.isExpanded ? <Minimize size={12} /> : <Eye size={12} />}
            </button>

            <div className="font-semibold text-sm">{layer.name}</div>
            <div className="text-xs text-muted-foreground mt-1">{layer.type}</div>
            <div className="text-xs mt-1">
                Shape: {layer.shape?.join("×") || "Unknown"}
            </div>

            {layer.isExpanded && (
                <div className="mt-2 text-xs bg-black/10 rounded p-2">
                    <div>Units: {layer.shape?.[layer.shape.length - 1] || 'Unknown'}</div>
                    <div>Activation: {layer.activationMagnitude?.toFixed(4) || "0"}</div>
                </div>
            )}

            {/* Output handle - right side */}
            <Handle
                type="source"
                position={Position.Right}
                id="output"
                style={{ background: '#555', width: 8, height: 8 }}
            />

            {/* Tooltip content - shown on hover */}
            <div className="absolute top-full left-1/2 transform -translate-x-1/2 mt-2 
                      bg-popover text-popover-foreground p-3 rounded-md shadow-lg 
                      border opacity-0 hover:opacity-100 transition-opacity z-50
                      min-w-[200px] text-left pointer-events-none">
                <div className="font-semibold mb-2">{layer.name}</div>
                <div className="space-y-1 text-xs">
                    <div>Type: {layer.type}</div>
                    <div>Shape: {layer.shape?.join("×") || "Unknown"}</div>
                    <div>Activation Magnitude: {layer.activationMagnitude?.toFixed(4) || "0"}</div>
                    {layer.isInput && <div className="text-blue-600 font-semibold">Input Layer</div>}
                    {layer.isOutput && <div className="text-purple-600 font-semibold">Output Layer</div>}
                    {stats && (
                        <>
                            <div>Mean: {stats.mean.toFixed(4)}</div>
                            <div>Max: {stats.max.toFixed(4)}</div>
                            <div>Min: {stats.min.toFixed(4)}</div>
                            <div>Std: {stats.std.toFixed(4)}</div>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
};

// Custom Neuron Node Component
const NeuronNode: React.FC<{ data: NeuronNodeData }> = ({ data }) => {
    const { neuron } = data;

    // Calculate color intensity based on activation
    const intensity = Math.min(1, Math.abs(neuron.activation));
    const hue = neuron.activation >= 0 ? 120 : 0;

    return (
        <div
            className={cn(
                "w-8 h-8 rounded-full border-2 border-muted-foreground/20",
                "shadow-md hover:shadow-lg transition-all duration-200",
                "flex items-center justify-center text-xs font-semibold relative"
            )}
            style={{
                backgroundColor: `hsla(${hue}, 70%, 50%, ${intensity * 0.5 + 0.1})`,
                borderColor: `hsla(${hue}, 70%, 40%, ${intensity * 0.7 + 0.3})`,
                color: intensity > 0.5 ? "white" : "black",
            }}
            title={`Neuron ${neuron.index}: ${neuron.activation.toFixed(4)}`}
        >
            {/* Input handle - left side */}
            <Handle
                type="target"
                position={Position.Left}
                id="input"
                style={{ background: '#555', width: 6, height: 6, left: -3 }}
            />

            {neuron.index}

            {/* Output handle - right side */}
            <Handle
                type="source"
                position={Position.Right}
                id="output"
                style={{ background: '#555', width: 6, height: 6, right: -3 }}
            />
        </div>
    );
};

// Custom Edge Component
const WeightEdge: React.FC<WeightEdgeProps> = ({
    id,
    sourceX,
    sourceY,
    targetX,
    targetY,
    data
}) => {
    const { weight, thickness } = data;

    const edgePath = `M${sourceX},${sourceY} L${targetX},${targetY}`;
    const isPositive = weight >= 0;

    return (
        <g>
            <path
                id={id}
                style={{
                    stroke: isPositive ? "#10b981" : "#ef4444",
                    strokeWidth: thickness,
                    strokeOpacity: 0.6,
                }}
                className="react-flow__edge-path"
                d={edgePath}
            />
            {Math.abs(weight) > 0.5 && (
                <text
                    className="text-xs fill-muted-foreground"
                    textAnchor="middle"
                    x={(sourceX + targetX) / 2}
                    y={(sourceY + targetY) / 2}
                >
                    {weight.toFixed(2)}
                </text>
            )}
        </g>
    );
};

const nodeTypes: NodeTypes = {
    layer: LayerNode,
    neuron: NeuronNode,
};

const edgeTypes: EdgeTypes = {
    weight: WeightEdge,
};

interface NetworkVisualizationProps {
    className?: string;
}

export const NetworkVisualization: React.FC<NetworkVisualizationProps> = ({
    className
}) => {
    const {
        viewMode,
        layerNodes,
        neuronNodes,
        edges,
        focusedLayerId,
        zoomLevel,
        expandedLayers,
        setFocusedLayer,
        collapseAllLayers,
    } = useModelStore();

    // Convert store data to React Flow format
    const nodes: Node[] = useMemo(() => {
        console.log("NetworkVisualization nodes calculation:", {
            viewMode,
            layerNodesCount: layerNodes.length,
            neuronNodesCount: neuronNodes.length,
            expandedLayersCount: expandedLayers.size
        });

        // Always include layer nodes
        const allNodes: Node[] = [];

        // If no layer nodes from store, create a test node to verify React Flow is working
        if (layerNodes.length === 0) {
            console.log("No layer nodes available, using test node fallback");
            return [{
                id: 'test-node',
                type: 'layer',
                position: { x: 200, y: 100 },
                data: {
                    layer: {
                        id: 'test-layer',
                        name: 'Test Layer',
                        type: 'Dense',
                        activationMagnitude: 0.5,
                        shape: [1, 10],
                        position: { x: 200, y: 100 },
                        stats: { mean: 0.1, max: 1, min: -1, std: 0.3 }
                    }
                }
            }];
        }

        // Add layer nodes
        layerNodes.forEach((node) => {
            allNodes.push({
                id: node.id,
                type: "layer",
                position: node.position,
                data: {
                    layer: node,
                    stats: node.stats,
                },
                style: {
                    opacity: focusedLayerId
                        ? node.id === focusedLayerId ? 1 : 0.3
                        : 1,
                },
            });
        });

        // Add neuron nodes for expanded layers
        if (expandedLayers.size > 0) {
            neuronNodes.forEach((node) => {
                allNodes.push({
                    id: node.id,
                    type: "neuron",
                    position: node.position,
                    data: {
                        neuron: node,
                    },
                    style: {
                        opacity: focusedLayerId
                            ? node.layerId === focusedLayerId ? 1 : 0.3
                            : 1,
                    },
                });
            });
        }

        console.log("Using combined nodes for visualization:", allNodes.length);
        return allNodes;
    }, [layerNodes, neuronNodes, focusedLayerId, expandedLayers]);

    const flowEdges: Edge[] = useMemo(() => {
        console.log("NetworkVisualization edges calculation:", {
            edgesCount: edges.length,
            edges: edges.slice(0, 3) // Log first 3 for debugging
        });

        return edges.map((edge) => ({
            id: edge.id,
            source: edge.source,
            target: edge.target,
            sourceHandle: "output", // Specify the handle ID
            targetHandle: "input",  // Specify the handle ID
            type: "weight",
            data: {
                weight: edge.weight,
                thickness: edge.thickness,
            },
            animated: edge.animated || false,
            style: {
                strokeWidth: edge.thickness,
                opacity: focusedLayerId ? 0.3 : 0.6,
            },
        }));
    }, [edges, focusedLayerId]);

    const [nodesState, setNodes, onNodesChange] = useNodesState(nodes);
    const [edgesState, setEdges, onEdgesChange] = useEdgesState(flowEdges);

    // Update nodes and edges when store changes
    React.useEffect(() => {
        setNodes(nodes);
    }, [nodes, setNodes]);

    React.useEffect(() => {
        setEdges(flowEdges);
    }, [flowEdges, setEdges]);

    const onConnect = useCallback(
        (params: Connection) => setEdges((eds) => addEdge(params, eds)),
        [setEdges]
    );

    const onNodeClick = useCallback(
        (_event: React.MouseEvent, node: Node) => {
            if (viewMode === "layer") {
                setFocusedLayer(focusedLayerId === node.id ? null : node.id);
            } else {
                const neuronData = node.data as unknown as NeuronNodeData;
                if (neuronData.neuron && "layerId" in neuronData.neuron) {
                    setFocusedLayer(
                        focusedLayerId === neuronData.neuron.layerId
                            ? null
                            : neuronData.neuron.layerId
                    );
                }
            }
        },
        [viewMode, focusedLayerId, setFocusedLayer]
    );

    return (
        <div className={cn("w-full h-full bg-background", className)}>
            {nodes.length === 0 ? (
                <div className="flex items-center justify-center h-full text-muted-foreground">
                    <div className="text-center">
                        <div className="text-lg font-medium mb-2">No Model Loaded</div>
                        <div className="text-sm">
                            Configure layers in the Hyperparameters tab to see the neural network visualization
                        </div>
                    </div>
                </div>
            ) : (
                <ReactFlow
                    nodes={nodesState}
                    edges={edgesState}
                    onNodesChange={onNodesChange}
                    onEdgesChange={onEdgesChange}
                    onConnect={onConnect}
                    onNodeClick={onNodeClick}
                    nodeTypes={nodeTypes}
                    edgeTypes={edgeTypes}
                    fitView
                    attributionPosition="bottom-left"
                    defaultViewport={{ x: 0, y: 0, zoom: zoomLevel }}
                    minZoom={0.1}
                    maxZoom={2}
                >
                    <Background color="#aaa" gap={16} />
                    <Controls />

                    {/* Custom Control Panel */}
                    <Panel position="top-right" className="bg-background border border-border rounded-lg p-3 m-2">
                        <div className="space-y-2">
                            <div className="text-sm font-medium">Layer Controls</div>
                            <div className="flex flex-col gap-2">
                                <Button
                                    size="sm"
                                    variant="outline"
                                    onClick={collapseAllLayers}
                                    className="text-xs"
                                >
                                    <EyeOff size={12} className="mr-1" />
                                    Collapse All
                                </Button>
                                <div className="text-xs text-muted-foreground">
                                    Expanded: {expandedLayers.size}
                                </div>
                            </div>
                        </div>
                    </Panel>

                    <MiniMap
                        className="bg-background border border-border"
                        nodeColor={(node) => {
                            if (node.type === "layer") {
                                const layerData = node.data as unknown as LayerNodeData;
                                const layer = layerData.layer;

                                if (layer.isInput) return "#3b82f6"; // Blue for input
                                if (layer.isOutput) return "#8b5cf6"; // Purple for output

                                const intensity = Math.min(1, Math.abs(layer.activationMagnitude) / 10);
                                const hue = layer.activationMagnitude >= 0 ? 120 : 0;
                                return `hsla(${hue}, 70%, 50%, ${intensity * 0.5 + 0.2})`;
                            }
                            return "#6366f1";
                        }}
                    />
                </ReactFlow>
            )}
        </div>
    );
};
