import React, { useState } from "react";
import { Button } from "./ui/button";
import { Slider } from "./ui/slider";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Play, Pause, Square, SkipForward, StepForward } from "lucide-react";
import { cn } from "../lib/utils";

export type TrainingState = "idle" | "training" | "paused" | "stopped";
export type TrainingSpeed = number; // 0 = manual, 1 = real-time, up to 10 = 10x speed

export interface MetricData {
    epoch: number;
    loss: number;
    accuracy?: number;
    valLoss?: number;
    valAccuracy?: number;
    timestamp: number;
}

interface TrainingControlsProps {
    trainingState: TrainingState;
    speed: TrainingSpeed;
    currentEpoch: number;
    totalEpochs: number;
    currentBatch: number;
    totalBatches: number;
    trainingMetrics: MetricData[];
    onStart: () => void;
    onPause: () => void;
    onStop: () => void;
    onSpeedChange: (speed: TrainingSpeed) => void;
    onNextBatch: () => void;
    onNextEpoch: () => void;
    onEpochScrub: (epoch: number) => void;
    canStep: boolean;
}

export const TrainingControls: React.FC<TrainingControlsProps> = ({
    trainingState,
    speed,
    currentEpoch,
    totalEpochs,
    currentBatch,
    totalBatches,
    trainingMetrics,
    onStart,
    onPause,
    onStop,
    onSpeedChange,
    onNextBatch,
    onNextEpoch,
    onEpochScrub,
    canStep,
}) => {
    const [scrubberEpoch, setScrubberEpoch] = useState(currentEpoch);

    const isTraining = trainingState === "training";
    const isPaused = trainingState === "paused";
    const isManualMode = speed === 0;

    // Use training metrics to determine actual completed epochs
    const completedEpochs = trainingMetrics.length;
    const displayEpoch = Math.max(completedEpochs, 0);

    const handleSpeedChange = (value: number[]) => {
        onSpeedChange(value[0]);
    };

    const handleEpochScrub = (value: number[]) => {
        const epoch = value[0];
        setScrubberEpoch(epoch);
        onEpochScrub(epoch);
    };

    const getSpeedLabel = (speed: number) => {
        if (speed === 0) return "Manual";
        if (speed === 1) return "Real-time";
        return `${speed}×`;
    };

    return (
        <Card className="w-full">
            <CardHeader>
                <CardTitle className="flex items-center justify-between">
                    Training Controls
                    <Badge variant={isTraining ? "default" : isPaused ? "secondary" : "outline"}>
                        {trainingState.charAt(0).toUpperCase() + trainingState.slice(1)}
                    </Badge>
                </CardTitle>
                <CardDescription>
                    Control the training process with real-time visualization
                </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
                {/* Main Controls */}
                <div className="flex items-center gap-3">
                    <Button
                        onClick={isTraining ? onPause : onStart}
                        disabled={trainingState === "stopped"}
                        className="flex-1"
                        size="lg"
                    >
                        {isTraining ? (
                            <>
                                <Pause className="mr-2 h-4 w-4" />
                                Pause
                            </>
                        ) : (
                            <>
                                <Play className="mr-2 h-4 w-4" />
                                {isPaused ? "Resume" : "Start Training"}
                            </>
                        )}
                    </Button>
                    <Button
                        onClick={onStop}
                        variant="destructive"
                        disabled={trainingState === "idle" || trainingState === "stopped"}
                    >
                        <Square className="mr-2 h-4 w-4" />
                        Stop
                    </Button>
                </div>

                {/* Speed Control */}
                <div className="space-y-3">
                    <div className="flex items-center justify-between">
                        <label className="text-sm font-medium">Training Speed</label>
                        <Badge variant="outline">{getSpeedLabel(speed)}</Badge>
                    </div>
                    <Slider
                        value={[speed]}
                        onValueChange={handleSpeedChange}
                        max={10}
                        min={0}
                        step={0.5}
                        className="w-full"
                        disabled={isTraining}
                    />
                    <div className="flex justify-between text-xs text-muted-foreground">
                        <span>Manual</span>
                        <span>1× Real-time</span>
                        <span>10× Fast</span>
                    </div>
                </div>

                {/* Manual Step Controls */}
                {isManualMode && (
                    <div className="space-y-3">
                        <label className="text-sm font-medium">Manual Stepping</label>
                        <div className="flex gap-2">
                            <Button
                                onClick={onNextBatch}
                                disabled={!canStep || isTraining}
                                variant="outline"
                                className="flex-1"
                            >
                                <StepForward className="mr-2 h-4 w-4" />
                                Next Batch
                            </Button>
                            <Button
                                onClick={onNextEpoch}
                                disabled={!canStep || isTraining}
                                variant="outline"
                                className="flex-1"
                            >
                                <SkipForward className="mr-2 h-4 w-4" />
                                Next Epoch
                            </Button>
                        </div>
                    </div>
                )}

                {/* Progress Display */}
                <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                        <span>Epoch Progress</span>
                        <span>{completedEpochs} / {totalEpochs}</span>
                    </div>
                    <div className="w-full bg-secondary rounded-full h-2">
                        <div
                            className="bg-primary h-2 rounded-full transition-all duration-300"
                            style={{
                                width: totalEpochs > 0 ? `${(completedEpochs / totalEpochs) * 100}%` : "0%"
                            }}
                        />
                    </div>
                    <div className="flex justify-between text-xs text-muted-foreground">
                        <span>Batch: {currentBatch} / {totalBatches}</span>
                        <span>{totalEpochs > 0 ? Math.round((completedEpochs / totalEpochs) * 100) : 0}%</span>
                    </div>
                </div>

                {/* Epoch Scrubber */}
                {completedEpochs > 0 && (
                    <div className="space-y-3">
                        <div className="flex items-center justify-between">
                            <label className="text-sm font-medium">Epoch Scrubber</label>
                            <Badge variant="outline">Epoch {scrubberEpoch + 1}</Badge>
                        </div>
                        <Slider
                            value={[scrubberEpoch]}
                            onValueChange={handleEpochScrub}
                            max={completedEpochs - 1}
                            min={0}
                            step={1}
                            className="w-full"
                            disabled={isTraining}
                        />
                        <div className="text-xs text-muted-foreground">
                            Replay visualization from any completed epoch
                        </div>
                    </div>
                )}
            </CardContent>
        </Card>
    );
};
