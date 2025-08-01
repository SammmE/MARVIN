import { useDataStore } from "@/lib/oscar-store";
import type { ChartType, PresetType } from "@/lib/oscar-store";
import { Button } from "./ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "./ui/card";
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
    DropdownMenuLabel,
    DropdownMenuSeparator,
} from "./ui/dropdown-menu";
import { ChevronDown, LineChart, BarChart, Trash2 } from "lucide-react";
import { FileUploader } from "./file-uploader";

// Custom icon for scatter plot
const ScatterPlotIcon = () => (
    <svg
        xmlns="http://www.w3.org/2000/svg"
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
    >
        <circle cx="7" cy="14" r="1" />
        <circle cx="12" cy="8" r="1" />
        <circle cx="16" cy="12" r="1" />
        <circle cx="18" cy="16" r="1" />
        <circle cx="10" cy="19" r="1" />
        <circle cx="5" cy="6" r="1" />
    </svg>
);

const chartOptions: {
    value: ChartType;
    label: string;
    icon: React.ReactNode;
}[] = [
        { value: "scatter", label: "Scatter Plot", icon: <ScatterPlotIcon /> },
        { value: "line", label: "Line Chart", icon: <LineChart size={16} /> },
        { value: "histogram", label: "Histogram", icon: <BarChart size={16} /> },
    ];

const presetOptions: { value: PresetType; label: string }[] = [
    { value: "sinusoid", label: "Sinusoid" },
    { value: "spiral", label: "Spiral" },
    { value: "moons", label: "Moons" },
    { value: "iris", label: "Iris" },
    { value: "mnist-subset", label: "MNIST Subset" },
];

export function DataControls() {
    const {
        dataset,
        chartType,
        xAxis,
        yAxis,
        selectedFeatures,
        selectedTarget,
        setChartType,
        setXAxis,
        setYAxis,
        setSelectedFeatures,
        setSelectedTarget,
        loadPreset,
        clearData,
    } = useDataStore();

    const handleFeatureToggle = (column: string) => {
        if (selectedFeatures.includes(column)) {
            setSelectedFeatures(selectedFeatures.filter((f) => f !== column));
        } else {
            setSelectedFeatures([...selectedFeatures, column]);
        }
    };

    const handleTargetSelect = (column: string) => {
        setSelectedTarget(column === selectedTarget ? null : column);
    };

    const handlePresetSelect = (preset: PresetType) => {
        if (preset) {
            loadPreset(preset);
        }
    };

    return (
        <div className="space-y-6">
            <Card>
                <CardHeader>
                    <CardTitle>Data Source</CardTitle>
                </CardHeader>
                <CardContent>
                    {!dataset ? (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <FileUploader />

                            <div className="space-y-4">
                                <h3 className="text-sm font-medium">Choose Preset</h3>
                                <div className="grid grid-cols-1 gap-2">
                                    {presetOptions.map((preset) => (
                                        <Button
                                            key={preset.value}
                                            variant="outline"
                                            className="justify-start"
                                            onClick={() => handlePresetSelect(preset.value)}
                                        >
                                            {preset.label}
                                        </Button>
                                    ))}
                                </div>
                            </div>
                        </div>
                    ) : (
                        <div className="space-y-4">
                            <Button
                                variant="outline"
                                onClick={clearData}
                                className="w-full"
                            >
                                <Trash2 className="mr-2 h-4 w-4" />
                                Clear Data
                            </Button>
                        </div>
                    )}
                </CardContent>
            </Card>

            {dataset && (
                <>
                    <Card>
                        <CardHeader>
                            <CardTitle>Visualization Settings</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div className="space-y-4">
                                    <div>
                                        <label className="text-sm font-medium">Chart Type</label>
                                        <DropdownMenu>
                                            <DropdownMenuTrigger asChild>
                                                <Button
                                                    variant="outline"
                                                    className="w-full justify-between mt-1"
                                                >
                                                    <div className="flex items-center">
                                                        {
                                                            chartOptions.find(
                                                                (option) => option.value === chartType,
                                                            )?.icon
                                                        }
                                                        <span className="ml-2">
                                                            {
                                                                chartOptions.find(
                                                                    (option) => option.value === chartType,
                                                                )?.label
                                                            }
                                                        </span>
                                                    </div>
                                                    <ChevronDown size={16} />
                                                </Button>
                                            </DropdownMenuTrigger>
                                            <DropdownMenuContent className="w-56">
                                                <DropdownMenuLabel>Chart Type</DropdownMenuLabel>
                                                <DropdownMenuSeparator />
                                                {chartOptions.map((option) => (
                                                    <DropdownMenuItem
                                                        key={option.value}
                                                        onClick={() => setChartType(option.value)}
                                                        className="flex items-center"
                                                    >
                                                        {option.icon}
                                                        <span className="ml-2">{option.label}</span>
                                                    </DropdownMenuItem>
                                                ))}
                                            </DropdownMenuContent>
                                        </DropdownMenu>
                                    </div>

                                    <div>
                                        <label className="text-sm font-medium">X-Axis</label>
                                        <DropdownMenu>
                                            <DropdownMenuTrigger asChild>
                                                <Button
                                                    variant="outline"
                                                    className="w-full justify-between mt-1"
                                                >
                                                    {xAxis || "Select X-Axis"}
                                                    <ChevronDown size={16} />
                                                </Button>
                                            </DropdownMenuTrigger>
                                            <DropdownMenuContent className="w-56">
                                                <DropdownMenuLabel>Select X-Axis</DropdownMenuLabel>
                                                <DropdownMenuSeparator />
                                                {dataset.columns.map((column) => (
                                                    <DropdownMenuItem
                                                        key={column}
                                                        onClick={() => setXAxis(column)}
                                                    >
                                                        {column}
                                                    </DropdownMenuItem>
                                                ))}
                                            </DropdownMenuContent>
                                        </DropdownMenu>
                                    </div>

                                    <div>
                                        <label className="text-sm font-medium">Y-Axis</label>
                                        <DropdownMenu>
                                            <DropdownMenuTrigger asChild>
                                                <Button
                                                    variant="outline"
                                                    className="w-full justify-between mt-1"
                                                >
                                                    {yAxis || "Select Y-Axis"}
                                                    <ChevronDown size={16} />
                                                </Button>
                                            </DropdownMenuTrigger>
                                            <DropdownMenuContent className="w-56">
                                                <DropdownMenuLabel>Select Y-Axis</DropdownMenuLabel>
                                                <DropdownMenuSeparator />
                                                {dataset.columns.map((column) => (
                                                    <DropdownMenuItem
                                                        key={column}
                                                        onClick={() => setYAxis(column)}
                                                    >
                                                        {column}
                                                    </DropdownMenuItem>
                                                ))}
                                            </DropdownMenuContent>
                                        </DropdownMenu>
                                    </div>
                                </div>

                                <div className="space-y-4">
                                    <div>
                                        <h3 className="text-sm font-medium mb-2">Features</h3>
                                        <div className="border rounded-md p-2 max-h-48 overflow-y-auto">
                                            {dataset.columns.map((column) => (
                                                <div key={column} className="flex items-center py-1">
                                                    <input
                                                        type="checkbox"
                                                        id={`feature-${column}`}
                                                        checked={selectedFeatures.includes(column)}
                                                        onChange={() => handleFeatureToggle(column)}
                                                        className="h-4 w-4 rounded border-gray-300 text-primary focus:ring-primary mr-2"
                                                    />
                                                    <label
                                                        htmlFor={`feature-${column}`}
                                                        className="text-sm"
                                                    >
                                                        {column}
                                                    </label>
                                                </div>
                                            ))}
                                        </div>
                                    </div>

                                    <div>
                                        <h3 className="text-sm font-medium mb-2">Target</h3>
                                        <div className="border rounded-md p-2 max-h-48 overflow-y-auto">
                                            {dataset.columns.map((column) => (
                                                <div key={column} className="flex items-center py-1">
                                                    <input
                                                        type="radio"
                                                        id={`target-${column}`}
                                                        name="target"
                                                        checked={selectedTarget === column}
                                                        onChange={() => handleTargetSelect(column)}
                                                        className="h-4 w-4 border-gray-300 text-primary focus:ring-primary mr-2"
                                                    />
                                                    <label
                                                        htmlFor={`target-${column}`}
                                                        className="text-sm"
                                                    >
                                                        {column}
                                                    </label>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                </>
            )}
        </div>
    );
}
