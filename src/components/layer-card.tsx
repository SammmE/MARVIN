import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import type {
	Layer,
	LayerType,
	ActivationType,
} from "@/lib/hyperparameters-store";
import {
	useHyperparametersStore,
} from "@/lib/hyperparameters-store";
import {
	ChevronUp,
	ChevronDown,
	Trash2,
	Settings,
	PlusCircle,
} from "lucide-react";
import {
	Dialog,
	DialogContent,
	DialogHeader,
	DialogTitle,
} from "@/components/ui/dialog";
import Editor from "@monaco-editor/react";

interface LayerCardProps {
	layer: Layer;
	isFirst: boolean;
	isLast: boolean;
}

export function LayerCard({ layer, isFirst, isLast }: LayerCardProps) {
	const [isEditing, setIsEditing] = useState(false);
	const [editingFunction, setEditingFunction] = useState<string | null>(null);
	const [functionName, setFunctionName] = useState<string>("");
	const [customCode, setCustomCode] = useState<string>(
		layer.customActivation || "x => Math.max(0, x) // ReLU example",
	);
	const [makeDefault, setMakeDefault] = useState(false);

	const updateLayer = useHyperparametersStore((state) => state.updateLayer);
	const removeLayer = useHyperparametersStore((state) => state.removeLayer);
	const moveLayerUp = useHyperparametersStore((state) => state.moveLayerUp);
	const moveLayerDown = useHyperparametersStore((state) => state.moveLayerDown);
	const setDefaultCustomFunction = useHyperparametersStore((state) => state.setDefaultCustomFunction);
	const globalCustomFunctions = useHyperparametersStore((state) => state.globalCustomFunctions);
	const addGlobalCustomFunction = useHyperparametersStore((state) => state.addGlobalCustomFunction);
	const updateGlobalCustomFunction = useHyperparametersStore((state) => state.updateGlobalCustomFunction);
	const removeGlobalCustomFunction = useHyperparametersStore((state) => state.removeGlobalCustomFunction);

	// Use global functions as the available custom functions
	const customFunctions = globalCustomFunctions;

	const handleTypeChange = (type: string) => {
		updateLayer(layer.id, { type: type as LayerType });
	};

	const handleActivationChange = (activation: string) => {
		// If activation is one of the custom functions, set the selectedCustomFunction
		if (customFunctions.some(f => f.name === activation)) {
			updateLayer(layer.id, { 
				activation: "Custom", 
				selectedCustomFunction: activation 
			});
		} else {
			// Otherwise it's one of the built-in activations
			updateLayer(layer.id, { 
				activation: activation as ActivationType,
				selectedCustomFunction: undefined
			});
		}
	};

	// Handle adding a new function
	const handleAddFunction = () => {
		setEditingFunction(null);
		setFunctionName("");
		setCustomCode(
			"function NewFunction(x) {\n  return x > 0 ? x : 0.01 * x;\n}",
		);
		setIsEditing(true);
		
		// Make sure the activation is set to Custom when adding a new function
		if (layer.activation !== "Custom") {
			updateLayer(layer.id, { activation: "Custom" });
		}
	};

	// Handle editing an existing function
	const handleEditFunction = (name: string) => {
		const func = customFunctions.find((f) => f.name === name);
		if (func) {
			setEditingFunction(name);
			setFunctionName(name);
			setCustomCode(func.code);
			setIsEditing(true);
		}
	};

	// Handle removing a function
	const handleRemoveFunction = (name: string) => {
		removeGlobalCustomFunction(name);
		// If this layer was using the removed function, reset it
		if (layer.selectedCustomFunction === name) {
			updateLayer(layer.id, {
				activation: "ReLU", // Reset to default
				selectedCustomFunction: undefined,
			});
		}
	};

	// Handle saving a custom function
	const handleSaveCustomCode = () => {
		// Extract function name from the code if not provided
		let extractedName = functionName;
		if (!extractedName) {
			const nameMatch = customCode.match(/function\s+([a-zA-Z0-9_]+)/);
			if (nameMatch && nameMatch[1]) {
				extractedName = nameMatch[1];
			} else {
				extractedName = "CustomFunction" + Date.now();
			}
		}

		if (editingFunction) {
			// Update existing global function
			updateGlobalCustomFunction(extractedName, customCode);
		} else {
			// Add new global function
			addGlobalCustomFunction(extractedName, customCode);
		}

		// Update the layer to use this function
		updateLayer(layer.id, {
			activation: "Custom", // Always set to Custom when saving a function
			selectedCustomFunction: extractedName,
			// For backward compatibility
			customActivation: customCode,
		});

		// If makeDefault is checked, set this as the default custom function
		if (makeDefault) {
			setDefaultCustomFunction(extractedName);
		}

		// Reset states
		setIsEditing(false);
		setMakeDefault(false);
	};

	const getLayerParams = () => {
		switch (layer.type) {
			case "Dense":
				return (
					<div className="grid grid-cols-2 gap-2">
						<div>
							<label className="text-xs text-muted-foreground">Units</label>
							<Input
								type="number"
								value={layer.units || 0}
								onChange={(e) =>
									updateLayer(layer.id, {
										units: parseInt(e.target.value) || 0,
									})
								}
								min={1}
							/>
						</div>
						<div>
							<label className="text-xs text-muted-foreground">
								Activation
							</label>
							<Select
								value={layer.activation === "Custom" && layer.selectedCustomFunction ? layer.selectedCustomFunction : (layer.activation || "ReLU")}
								onValueChange={handleActivationChange}
							>
								<SelectTrigger>
									<SelectValue placeholder="Activation" />
								</SelectTrigger>
								<SelectContent>
									<SelectItem value="ReLU">ReLU</SelectItem>
									<SelectItem value="Sigmoid">Sigmoid</SelectItem>
									<SelectItem value="Tanh">Tanh</SelectItem>
									
									{/* If we have custom functions, list them */}
									{customFunctions.length > 0 && (
										<>
											<SelectItem value="divider" disabled>
												<div className="h-px bg-muted my-1" />
											</SelectItem>
											<SelectItem value="Custom-Functions" disabled>
												<span className="text-xs text-muted-foreground">Custom Functions</span>
											</SelectItem>
											{customFunctions.map(func => (
												<SelectItem key={func.name} value={func.name}>
													{func.name}
												</SelectItem>
											))}
										</>
									)}
									
									{/* Add Custom Option */}
									<SelectItem value="divider2" disabled>
										<div className="h-px bg-muted my-1" />
									</SelectItem>
									<SelectItem value="Custom">+ Create New Function</SelectItem>
								</SelectContent>
							</Select>
						</div>
					</div>
				);
			case "Conv1D":
			case "Conv2D":
				return (
					<div className="grid grid-cols-3 gap-2">
						<div>
							<label className="text-xs text-muted-foreground">Filters</label>
							<Input
								type="number"
								value={layer.filters || 0}
								onChange={(e) =>
									updateLayer(layer.id, {
										filters: parseInt(e.target.value) || 0,
									})
								}
								min={1}
							/>
						</div>
						<div>
							<label className="text-xs text-muted-foreground">
								Kernel Size
							</label>
							<Input
								type="number"
								value={layer.kernelSize || 3}
								onChange={(e) =>
									updateLayer(layer.id, {
										kernelSize: parseInt(e.target.value) || 3,
									})
								}
								min={1}
							/>
						</div>
						<div>
							<label className="text-xs text-muted-foreground">
								Activation
							</label>
							<Select
								value={layer.activation === "Custom" && layer.selectedCustomFunction ? layer.selectedCustomFunction : (layer.activation || "ReLU")}
								onValueChange={handleActivationChange}
							>
								<SelectTrigger>
									<SelectValue placeholder="Activation" />
								</SelectTrigger>
								<SelectContent>
									<SelectItem value="ReLU">ReLU</SelectItem>
									<SelectItem value="Sigmoid">Sigmoid</SelectItem>
									<SelectItem value="Tanh">Tanh</SelectItem>
									
									{/* If we have custom functions, list them */}
									{customFunctions.length > 0 && (
										<>
											<SelectItem value="divider" disabled>
												<div className="h-px bg-muted my-1" />
											</SelectItem>
											<SelectItem value="Custom-Functions" disabled>
												<span className="text-xs text-muted-foreground">Custom Functions</span>
											</SelectItem>
											{customFunctions.map(func => (
												<SelectItem key={func.name} value={func.name}>
													{func.name}
												</SelectItem>
											))}
										</>
									)}
									
									{/* Add Custom Option */}
									<SelectItem value="divider2" disabled>
										<div className="h-px bg-muted my-1" />
									</SelectItem>
									<SelectItem value="Custom">+ Create New Function</SelectItem>
								</SelectContent>
							</Select>
						</div>
					</div>
				);
			case "Flatten":
				return (
					<div className="text-sm text-center text-muted-foreground py-2">
						Flatten layer (no parameters)
					</div>
				);
			default:
				return null;
		}
	};

	return (
		<Card className="mb-4">
			<CardHeader className="pb-2 flex flex-row items-center justify-between">
				<div>
					<CardTitle className="text-sm font-medium">
						{layer.type} Layer
					</CardTitle>
					{layer.inputSize !== undefined && layer.outputSize !== undefined && (
						<div className="text-xs text-muted-foreground mt-1">
							{layer.inputSize} → {layer.outputSize} units
						</div>
					)}
				</div>
				<div className="flex items-center space-x-1">
					<Button
						variant="ghost"
						size="sm"
						disabled={isFirst}
						onClick={() => moveLayerUp(layer.id)}
					>
						<ChevronUp className="h-4 w-4" />
					</Button>
					<Button
						variant="ghost"
						size="sm"
						disabled={isLast}
						onClick={() => moveLayerDown(layer.id)}
					>
						<ChevronDown className="h-4 w-4" />
					</Button>
					<Button
						variant="ghost"
						size="sm"
						onClick={() => removeLayer(layer.id)}
					>
						<Trash2 className="h-4 w-4 text-red-500" />
					</Button>
				</div>
			</CardHeader>
			<CardContent>
				<div className="space-y-4">
					<div>
						<label className="text-xs text-muted-foreground">Layer Type</label>
						<Select value={layer.type} onValueChange={handleTypeChange}>
							<SelectTrigger>
								<SelectValue placeholder="Layer Type" />
							</SelectTrigger>
							<SelectContent>
								<SelectItem value="Dense">Dense</SelectItem>
								<SelectItem value="Conv1D">Conv1D</SelectItem>
								<SelectItem value="Conv2D">Conv2D</SelectItem>
								<SelectItem value="Flatten">Flatten</SelectItem>
							</SelectContent>
						</Select>
					</div>

					{getLayerParams()}

					{layer.activation === "Custom" && (
						<div className="mt-2">
							{/* Display the list of custom functions */}
							{customFunctions.length > 0 && (
								<div className="mb-2">
									<label className="text-xs text-muted-foreground mb-1 block">
										{layer.selectedCustomFunction ? "Selected Function:" : "Available Functions:"}
									</label>
									<div className="space-y-1">
										{customFunctions.map((func) => (
											<div
												key={func.name}
												className={`flex justify-between items-center p-2 rounded-md ${func.name === layer.selectedCustomFunction ? 'bg-primary/20 border border-primary/50' : 'bg-secondary/30'}`}
											>
												<div className="flex items-center">
													{func.name === layer.selectedCustomFunction && (
														<div className="w-2 h-2 bg-primary rounded-full mr-2"></div>
													)}
													<span className={`font-medium text-sm ${func.name === layer.selectedCustomFunction ? 'text-primary' : ''}`}>
														{func.name}
													</span>
												</div>
												<div className="flex items-center">
													{func.name !== layer.selectedCustomFunction && (
														<Button
															variant="ghost"
															size="sm"
															onClick={() => updateLayer(layer.id, { selectedCustomFunction: func.name })}
															className="mr-1"
														>
															<span className="text-xs">Select</span>
														</Button>
													)}
													<Button
														variant="ghost"
														size="sm"
														onClick={() => handleEditFunction(func.name)}
													>
														<Settings className="h-3 w-3" />
													</Button>
													<Button
														variant="ghost"
														size="sm"
														onClick={() => handleRemoveFunction(func.name)}
													>
														<Trash2 className="h-3 w-3 text-red-500" />
													</Button>
												</div>
											</div>
										))}
									</div>
								</div>
							)}

							{/* Button to add new function */}
							<Button
								variant="outline"
								className="w-full mb-2"
								onClick={handleAddFunction}
							>
								<PlusCircle className="mr-2 h-4 w-4" />
								Add Custom Function
							</Button>

							{/* Dialog for editing */}
							<Dialog open={isEditing} onOpenChange={setIsEditing}>
								<DialogContent className="sm:max-w-[600px] max-h-[80vh]">
									<DialogHeader>
										<DialogTitle>
											{editingFunction
												? `Edit ${editingFunction}`
												: "New Custom Function"}
										</DialogTitle>
									</DialogHeader>

									<div className="mb-3">
										<label className="text-sm font-medium">
											Function Name:
										</label>
										<Input
											value={functionName}
											onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
												setFunctionName(e.target.value)
											}
											placeholder="Enter function name or leave blank to extract from code"
										/>
									</div>

									<div className="h-[300px] border">
										<Editor
											height="100%"
											defaultLanguage="javascript"
											value={customCode}
											onChange={(value) => setCustomCode(value || "")}
											options={{
												minimap: { enabled: false },
												lineNumbers: "on",
												fontSize: 14,
											}}
										/>
									</div>
									<div className="flex justify-end space-x-2 mt-4">
										<Button
											variant="outline"
											onClick={() => setIsEditing(false)}
										>
											Cancel
										</Button>
										<Button onClick={handleSaveCustomCode}>
											{editingFunction ? "Update" : "Add"} Function
										</Button>
									</div>
								</DialogContent>
							</Dialog>
						</div>
					)}
				</div>
			</CardContent>
		</Card>
	);
}
