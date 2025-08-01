import React, { useCallback, useState } from "react";
import html2canvas from "html2canvas";
import { useModelStore, type ViewMode } from "../lib/oscar-store";
import { Button } from "./ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Slider } from "./ui/slider";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "./ui/dialog";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import {
    Play,
    Pause,
    Camera,
    Layers,
    Zap,
    Download,
    Trash2,
    Settings,
    ZoomIn,
    ZoomOut,
    RotateCcw
} from "lucide-react";
import { cn } from "../lib/utils";

interface ModelControlsProps {
    className?: string;
    canvasRef?: React.RefObject<HTMLElement>;
}

export const ModelControls: React.FC<ModelControlsProps> = ({
    className,
    canvasRef
}) => {
    const {
        viewMode,
        playState,
        focusedLayerId,
        zoomLevel,
        liveUpdateEnabled,
        updateInterval,
        layers,
        snapshots,
        isTraining,
        currentEpoch,
        totalEpochs,
        currentMetrics,
        setViewMode,
        setPlayState,
        setFocusedLayer,
        setZoomLevel,
        toggleLiveUpdates,
        setUpdateInterval,
        createSnapshot,
        deleteSnapshot,
        clearSnapshots,
    } = useModelStore();

    const [isSnapshotDialogOpen, setIsSnapshotDialogOpen] = useState(false);
    const [isSettingsDialogOpen, setIsSettingsDialogOpen] = useState(false);

    // Handle play/pause
    const handlePlayPause = useCallback(() => {
        const newState = playState === "playing" ? "paused" : "playing";
        setPlayState(newState);

        if (newState === "playing") {
            // Resume live updates if they were enabled
            if (!liveUpdateEnabled) {
                toggleLiveUpdates();
            }
        } else {
            // Pause live updates
            if (liveUpdateEnabled) {
                toggleLiveUpdates();
            }
        }
    }, [playState, liveUpdateEnabled, setPlayState, toggleLiveUpdates]);

    // Handle view mode toggle
    const handleViewModeToggle = useCallback(() => {
        const newMode: ViewMode = viewMode === "layer" ? "neuron" : "layer";
        setViewMode(newMode);
    }, [viewMode, setViewMode]);

    // Handle snapshot creation
    const handleSnapshot = useCallback(async () => {
        try {
            const snapshotId = await createSnapshot();

            // If canvas ref is provided, try to capture the visual
            if (canvasRef?.current) {
                const canvas = await html2canvas(canvasRef.current, {
                    backgroundColor: null,
                    scale: 1,
                });

                const dataUrl = canvas.toDataURL("image/png");

                // Update the snapshot with the image data
                // Note: This would require extending the store to update snapshot data
                console.log("Snapshot created with ID:", snapshotId);
                console.log("Canvas data URL:", dataUrl);
            }
        } catch (error) {
            console.error("Error creating snapshot:", error);
        }
    }, [createSnapshot, canvasRef]);

    // Handle zoom
    const handleZoomIn = useCallback(() => {
        setZoomLevel(Math.min(2, zoomLevel * 1.2));
    }, [zoomLevel, setZoomLevel]);

    const handleZoomOut = useCallback(() => {
        setZoomLevel(Math.max(0.1, zoomLevel * 0.8));
    }, [zoomLevel, setZoomLevel]);

    const handleZoomReset = useCallback(() => {
        setZoomLevel(1);
    }, [setZoomLevel]);

    // Download snapshot as PNG
    const downloadSnapshot = useCallback(async (snapshot: typeof snapshots[0]) => {
        if (canvasRef?.current) {
            const canvas = await html2canvas(canvasRef.current);
            const link = document.createElement("a");
            link.download = `model-snapshot-epoch-${snapshot.epoch}-${snapshot.timestamp}.png`;
            link.href = canvas.toDataURL();
            link.click();
        }
    }, [canvasRef]);

    return (
        <div className={cn("flex flex-col gap-4 p-4 bg-card rounded-lg border", className)}>
            {/* Main Controls Row */}
            <div className="flex items-center gap-2 flex-wrap">
                {/* Play/Pause */}
                <Button
                    variant={playState === "playing" ? "default" : "outline"}
                    size="sm"
                    onClick={handlePlayPause}
                    disabled={!isTraining}
                    className="flex items-center gap-2"
                >
                    {playState === "playing" ? (
                        <>
                            <Pause className="h-4 w-4" />
                            Pause
                        </>
                    ) : (
                        <>
                            <Play className="h-4 w-4" />
                            Play
                        </>
                    )}
                </Button>

                {/* View Mode Toggle */}
                <Button
                    variant="outline"
                    size="sm"
                    onClick={handleViewModeToggle}
                    className="flex items-center gap-2"
                >
                    {viewMode === "layer" ? (
                        <>
                            <Layers className="h-4 w-4" />
                            Layer View
                        </>
                    ) : (
                        <>
                            <Zap className="h-4 w-4" />
                            Neuron View
                        </>
                    )}
                </Button>

                {/* Snapshot */}
                <Button
                    variant="outline"
                    size="sm"
                    onClick={handleSnapshot}
                    className="flex items-center gap-2"
                >
                    <Camera className="h-4 w-4" />
                    Snapshot
                </Button>

                {/* Zoom Controls */}
                <div className="flex items-center gap-1 border rounded-md">
                    <Button variant="ghost" size="sm" onClick={handleZoomOut}>
                        <ZoomOut className="h-4 w-4" />
                    </Button>
                    <span className="px-2 text-sm font-mono">
                        {Math.round(zoomLevel * 100)}%
                    </span>
                    <Button variant="ghost" size="sm" onClick={handleZoomIn}>
                        <ZoomIn className="h-4 w-4" />
                    </Button>
                    <Button variant="ghost" size="sm" onClick={handleZoomReset}>
                        <RotateCcw className="h-4 w-4" />
                    </Button>
                </div>

                {/* Settings */}
                <Dialog open={isSettingsDialogOpen} onOpenChange={setIsSettingsDialogOpen}>
                    <DialogTrigger asChild>
                        <Button variant="outline" size="sm">
                            <Settings className="h-4 w-4" />
                        </Button>
                    </DialogTrigger>
                    <DialogContent>
                        <DialogHeader>
                            <DialogTitle>Visualization Settings</DialogTitle>
                        </DialogHeader>
                        <div className="space-y-4">
                            <div className="space-y-2">
                                <label className="text-sm font-medium">
                                    Update Interval (ms)
                                </label>
                                <Slider
                                    value={[updateInterval]}
                                    onValueChange={(value) => setUpdateInterval(value[0])}
                                    min={50}
                                    max={1000}
                                    step={50}
                                    className="w-full"
                                />
                                <div className="text-xs text-muted-foreground">
                                    Current: {updateInterval}ms
                                </div>
                            </div>

                            <div className="flex items-center justify-between">
                                <label className="text-sm font-medium">
                                    Live Updates
                                </label>
                                <Button
                                    variant={liveUpdateEnabled ? "default" : "outline"}
                                    size="sm"
                                    onClick={toggleLiveUpdates}
                                >
                                    {liveUpdateEnabled ? "Enabled" : "Disabled"}
                                </Button>
                            </div>
                        </div>
                    </DialogContent>
                </Dialog>
            </div>

            {/* Focus Layer Selector */}
            {layers.length > 0 && (
                <div className="flex items-center gap-2">
                    <label className="text-sm font-medium whitespace-nowrap">
                        Focus Layer:
                    </label>
                    <Select
                        value={focusedLayerId || ""}
                        onValueChange={(value) => setFocusedLayer(value === "all" ? null : value)}
                    >
                        <SelectTrigger className="w-48">
                            <SelectValue placeholder="All layers" />
                        </SelectTrigger>
                        <SelectContent>
                            <SelectItem value="all">All layers</SelectItem>
                            {layers.map((layer) => (
                                <SelectItem key={layer.id} value={layer.id}>
                                    {layer.name} ({layer.type})
                                </SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                </div>
            )}

            {/* Training Status */}
            {isTraining && (
                <div className="flex items-center gap-4 text-sm">
                    <div className="flex items-center gap-2">
                        <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                        <span>Training</span>
                    </div>
                    <div>Epoch: {currentEpoch}/{totalEpochs}</div>
                    {currentMetrics && (
                        <>
                            <div>Loss: {currentMetrics.loss.toFixed(4)}</div>
                            {currentMetrics.accuracy && (
                                <div>Accuracy: {(currentMetrics.accuracy * 100).toFixed(2)}%</div>
                            )}
                        </>
                    )}
                </div>
            )}

            {/* Snapshots */}
            {snapshots.length > 0 && (
                <Dialog open={isSnapshotDialogOpen} onOpenChange={setIsSnapshotDialogOpen}>
                    <DialogTrigger asChild>
                        <Button variant="outline" size="sm" className="self-start">
                            View Snapshots ({snapshots.length})
                        </Button>
                    </DialogTrigger>
                    <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
                        <DialogHeader>
                            <DialogTitle>Model Snapshots</DialogTitle>
                            <div className="flex gap-2">
                                <Button
                                    variant="outline"
                                    size="sm"
                                    onClick={clearSnapshots}
                                    disabled={snapshots.length === 0}
                                >
                                    <Trash2 className="h-4 w-4 mr-2" />
                                    Clear All
                                </Button>
                            </div>
                        </DialogHeader>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                            {snapshots.map((snapshot) => (
                                <Card key={snapshot.id} className="relative">
                                    <CardHeader className="pb-2">
                                        <CardTitle className="text-sm">
                                            Epoch {snapshot.epoch}
                                        </CardTitle>
                                        <CardDescription className="text-xs">
                                            {new Date(snapshot.timestamp).toLocaleString()}
                                        </CardDescription>
                                    </CardHeader>
                                    <CardContent className="space-y-2">
                                        <div className="text-xs space-y-1">
                                            <div>Loss: {snapshot.loss.toFixed(4)}</div>
                                            {snapshot.accuracy && (
                                                <div>Accuracy: {(snapshot.accuracy * 100).toFixed(2)}%</div>
                                            )}
                                            <div>Layers: {snapshot.layers.length}</div>
                                        </div>
                                        <div className="flex gap-1">
                                            <Button
                                                variant="outline"
                                                size="sm"
                                                onClick={() => downloadSnapshot(snapshot)}
                                                className="flex-1"
                                            >
                                                <Download className="h-3 w-3 mr-1" />
                                                Download
                                            </Button>
                                            <Button
                                                variant="outline"
                                                size="sm"
                                                onClick={() => deleteSnapshot(snapshot.id)}
                                            >
                                                <Trash2 className="h-3 w-3" />
                                            </Button>
                                        </div>
                                    </CardContent>
                                </Card>
                            ))}
                        </div>
                    </DialogContent>
                </Dialog>
            )}
        </div>
    );
};
