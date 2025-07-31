import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import { PlusCircle } from "lucide-react";
import { LayerCard } from "@/components/layer-card";
import { useHyperparametersStore } from "@/lib/hyperparameters-store";
import { ShadcnActivationFunctionGraph } from "@/components/shadcn-activation-function-graph";

export default function HyperparametersPage() {
    const {
        layers,
        addLayer,
        learningRate,
        batchSize,
        epochs,
        optimizer,
        totalParams,
        estimatedMemory,
        globalCustomFunctions,
        setLearningRate,
        setBatchSize,
        setEpochs,
        setOptimizer,
    } = useHyperparametersStore();

    // Define built-in activation functions
    const getBuiltinActivationFunctions = () => {
        return [
            {
                name: "ReLU",
                code: "function ReLU(x) {\n  return Math.max(0, x);\n}"
            },
            {
                name: "Sigmoid",
                code: "function Sigmoid(x) {\n  return 1 / (1 + Math.exp(-x));\n}"
            },
            {
                name: "Tanh",
                code: "function Tanh(x) {\n  return Math.tanh(x);\n}"
            }
        ];
    };

    // Collect all activation functions (built-in + global custom)
    const getAllActivationFunctions = () => {
        const functionMap = new Map<string, { name: string; code: string; selected?: boolean }>();

        // Add built-in functions first
        getBuiltinActivationFunctions().forEach(func => {
            functionMap.set(func.name, func);
        });

        // Add global custom functions
        globalCustomFunctions.forEach(func => {
            // Check if this function is selected in any layer
            const isSelected = layers.some(layer => 
                layer.activation === "Custom" && layer.selectedCustomFunction === func.name
            );
            
            functionMap.set(func.name, {
                ...func,
                selected: isSelected
            });
        });

        // Collect any legacy custom functions for backward compatibility
        layers.forEach(layer => {
            if (layer.customActivationFunctions && layer.customActivationFunctions.length > 0) {
                layer.customActivationFunctions.forEach(func => {
                    // Only add if not already in global functions
                    if (!functionMap.has(func.name)) {
                        const isSelected = layer.activation === "Custom" && layer.selectedCustomFunction === func.name;
                        functionMap.set(func.name, {
                            ...func,
                            selected: isSelected
                        });
                    }
                });
            } else if (layer.customActivation) {
                // Backward compatibility for single custom activation
                if (!functionMap.has("CustomActivation")) {
                    functionMap.set("CustomActivation", {
                        name: "CustomActivation",
                        code: layer.customActivation,
                        selected: layer.activation === "Custom"
                    });
                }
            }
        });

        // Convert map to array
        return Array.from(functionMap.values());
    };

    // Handler for adding a new layer with default values
    const handleAddLayer = () => {
        addLayer({
            type: "Dense",
            units: 64,
            activation: "ReLU",
        });
    };

    return (
        <div className="container mx-auto py-6">
            <h1 className="text-3xl font-bold mb-6">Model Hyperparameters</h1>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
                {/* Model Statistics */}
                <Card>
                    <CardHeader>
                        <CardTitle>Model Summary</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="space-y-2">
                            <div className="flex justify-between">
                                <span className="font-medium">Total Layers:</span>
                                <span>{layers.length}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="font-medium">Total Parameters:</span>
                                <span>{totalParams.toLocaleString()}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="font-medium">Input Size:</span>
                                <span>{layers[0]?.inputSize || 0}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="font-medium">Output Size:</span>
                                <span>{layers[layers.length - 1]?.outputSize || 0}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="font-medium">Est. Memory:</span>
                                <span>{estimatedMemory.toFixed(2)} MB</span>
                            </div>
                        </div>
                    </CardContent>
                </Card>

                {/* Training Settings */}
                <Card className="col-span-1 lg:col-span-2">
                    <CardHeader>
                        <CardTitle>Training Settings</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <div className="space-y-2">
                                <label htmlFor="learning-rate" className="text-sm font-medium">
                                    Learning Rate
                                </label>
                                <div className="flex items-center space-x-2">
                                    <Slider
                                        id="learning-rate"
                                        min={0.0001}
                                        max={0.1}
                                        step={0.0001}
                                        defaultValue={[learningRate]}
                                        onValueChange={([value]) => setLearningRate(value)}
                                    />
                                    <Input
                                        type="number"
                                        value={learningRate}
                                        onChange={(e) =>
                                            setLearningRate(parseFloat(e.target.value))
                                        }
                                        className="w-20"
                                    />
                                </div>
                            </div>

                            <div className="space-y-2">
                                <label htmlFor="batch-size" className="text-sm font-medium">
                                    Batch Size
                                </label>
                                <div className="space-y-2">
                                    <Select
                                        value={batchSize.toString()}
                                        onValueChange={(value) => {
                                            if (value === "Custom") {
                                                setBatchSize("Custom", 32);
                                            } else {
                                                setBatchSize(Number(value) as 16 | 32 | 64 | 128);
                                            }
                                        }}
                                    >
                                        <SelectTrigger>
                                            <SelectValue placeholder="Batch Size" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="16">16</SelectItem>
                                            <SelectItem value="32">32</SelectItem>
                                            <SelectItem value="64">64</SelectItem>
                                            <SelectItem value="128">128</SelectItem>
                                            <SelectItem value="Custom">Custom</SelectItem>
                                        </SelectContent>
                                    </Select>

                                    {batchSize === "Custom" && (
                                        <Input
                                            type="number"
                                            placeholder="Custom batch size"
                                            onChange={(e) =>
                                                setBatchSize("Custom", parseInt(e.target.value))
                                            }
                                            className="mt-2"
                                        />
                                    )}
                                </div>
                            </div>

                            <div className="space-y-2">
                                <label htmlFor="epochs" className="text-sm font-medium">
                                    Epochs
                                </label>
                                <Input
                                    id="epochs"
                                    type="number"
                                    value={epochs}
                                    onChange={(e) => setEpochs(parseInt(e.target.value))}
                                    className="w-full"
                                />
                            </div>

                            <div className="space-y-2">
                                <label htmlFor="optimizer" className="text-sm font-medium">
                                    Optimizer
                                </label>
                                <Select
                                    onValueChange={(value: string) =>
                                        setOptimizer(
                                            value as "SGD" | "Adam" | "RMSprop" | "Adagrad",
                                        )
                                    }
                                    defaultValue={optimizer}
                                >
                                    <SelectTrigger>
                                        <SelectValue placeholder="Select optimizer" />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="SGD">SGD</SelectItem>
                                        <SelectItem value="Adam">Adam</SelectItem>
                                        <SelectItem value="RMSprop">RMSprop</SelectItem>
                                        <SelectItem value="Adagrad">Adagrad</SelectItem>
                                    </SelectContent>
                                </Select>
                            </div>
                        </div>
                    </CardContent>
                </Card>
            </div>

            {/* Layer Configuration */}
            <Card>
                <CardHeader className="flex flex-row items-center justify-between">
                    <CardTitle>Network Architecture</CardTitle>
                    <Button
                        variant="outline"
                        size="sm"
                        onClick={handleAddLayer}
                        className="flex items-center gap-1"
                    >
                        <PlusCircle className="h-4 w-4" />
                        Add Layer
                    </Button>
                </CardHeader>
                <CardContent>
                    <div className="space-y-4">
                        {layers.map((layer, idx) => (
                            <LayerCard
                                key={layer.id}
                                layer={layer}
                                isFirst={idx === 0}
                                isLast={idx === layers.length - 1}
                            />
                        ))}

                        {layers.length === 0 && (
                            <div className="flex flex-col items-center justify-center p-8 border border-dashed rounded-lg">
                                <p className="text-muted-foreground mb-4">
                                    No layers defined yet
                                </p>
                                <Button
                                    variant="outline"
                                    onClick={handleAddLayer}
                                    className="flex items-center gap-1"
                                >
                                    <PlusCircle className="h-4 w-4" />
                                    Add First Layer
                                </Button>
                            </div>
                        )}
                    </div>
                </CardContent>
            </Card>

            {/* Activation Function Visualization */}
            <ShadcnActivationFunctionGraph
                functions={getAllActivationFunctions()}
                title="Activation Functions Visualization"
            />
        </div>
    );
}
