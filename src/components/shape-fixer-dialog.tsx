import React from "react";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "./ui/dialog";
import { Button } from "./ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Alert, AlertDescription } from "./ui/alert";
import { AlertTriangle, Zap, CheckCircle, Info } from "lucide-react";
import type { ShapeValidationResult, ShapeIssue, ShapeFix } from "../lib/shape-validator";

interface ShapeFixerDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    validationResult: ShapeValidationResult;
    onApplyFixes: (fixes: ShapeFix[]) => void;
    onDismiss: () => void;
}

const getIssueIcon = (type: ShapeIssue['type']) => {
    switch (type) {
        case 'input_mismatch':
            return <Info className="h-4 w-4" />;
        case 'layer_mismatch':
            return <AlertTriangle className="h-4 w-4" />;
        case 'output_mismatch':
            return <AlertTriangle className="h-4 w-4" />;
        default:
            return <AlertTriangle className="h-4 w-4" />;
    }
};

const getIssueColor = (type: ShapeIssue['type']) => {
    switch (type) {
        case 'input_mismatch':
            return 'bg-blue-100 text-blue-800';
        case 'layer_mismatch':
            return 'bg-yellow-100 text-yellow-800';
        case 'output_mismatch':
            return 'bg-red-100 text-red-800';
        default:
            return 'bg-gray-100 text-gray-800';
    }
};

const getIssueTypeLabel = (type: ShapeIssue['type']) => {
    switch (type) {
        case 'input_mismatch':
            return 'Input Shape';
        case 'layer_mismatch':
            return 'Layer Shape';
        case 'output_mismatch':
            return 'Output Shape';
        default:
            return 'Unknown';
    }
};

export const ShapeFixerDialog: React.FC<ShapeFixerDialogProps> = ({
    open,
    onOpenChange,
    validationResult,
    onApplyFixes,
    onDismiss,
}) => {
    const { issues, suggestions } = validationResult;
    const hasIssues = issues.length > 0;
    const hasFixes = suggestions.length > 0;

    const handleSmartFix = () => {
        onApplyFixes(suggestions);
        onOpenChange(false);
    };

    const handleDismiss = () => {
        onDismiss();
        onOpenChange(false);
    };

    if (!hasIssues) {
        return null;
    }

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
                <DialogHeader>
                    <DialogTitle className="flex items-center gap-2">
                        <AlertTriangle className="h-5 w-5 text-yellow-500" />
                        Shape Validation Issues Detected
                    </DialogTitle>
                    <DialogDescription>
                        The model architecture has shape compatibility issues that may prevent training or cause poor performance.
                    </DialogDescription>
                </DialogHeader>

                <div className="space-y-4">
                    {/* Issues Section */}
                    <div>
                        <h3 className="text-lg font-semibold mb-3">Issues Found</h3>
                        <div className="space-y-3">
                            {issues.map((issue, index) => (
                                <Card key={index} className="border-l-4 border-l-yellow-500">
                                    <CardHeader className="pb-2">
                                        <CardTitle className="text-sm flex items-center gap-2">
                                            {getIssueIcon(issue.type)}
                                            {issue.layerName}
                                            <Badge variant="secondary" className={getIssueColor(issue.type)}>
                                                {getIssueTypeLabel(issue.type)}
                                            </Badge>
                                        </CardTitle>
                                    </CardHeader>
                                    <CardContent className="pt-0">
                                        <p className="text-sm text-gray-600 mb-2">{issue.message}</p>
                                        <div className="flex gap-4 text-xs">
                                            <span>
                                                <strong>Expected:</strong> {Array.isArray(issue.expected) ? issue.expected.join(' × ') : issue.expected}
                                            </span>
                                            <span>
                                                <strong>Actual:</strong> {Array.isArray(issue.actual) ? issue.actual.join(' × ') : issue.actual}
                                            </span>
                                        </div>
                                    </CardContent>
                                </Card>
                            ))}
                        </div>
                    </div>

                    {/* Suggestions Section */}
                    {hasFixes && (
                        <div>
                            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                                <Zap className="h-5 w-5 text-blue-500" />
                                Smart Fix Suggestions
                            </h3>
                            <div className="space-y-2">
                                {suggestions.map((suggestion, index) => (
                                    <Alert key={index} className="border-l-4 border-l-blue-500">
                                        <CheckCircle className="h-4 w-4" />
                                        <AlertDescription>
                                            <strong>Layer {suggestion.layerIndex + 1}:</strong> {suggestion.description}
                                        </AlertDescription>
                                    </Alert>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Warning about proceeding without fixes */}
                    {!hasFixes && (
                        <Alert className="border-yellow-200 bg-yellow-50">
                            <AlertTriangle className="h-4 w-4" />
                            <AlertDescription>
                                No automatic fixes are available for these issues. You may need to manually adjust your model architecture before training.
                            </AlertDescription>
                        </Alert>
                    )}
                </div>

                <DialogFooter className="flex gap-2">
                    <Button variant="outline" onClick={handleDismiss}>
                        Dismiss
                    </Button>
                    {hasFixes && (
                        <Button onClick={handleSmartFix} className="bg-blue-600 hover:bg-blue-700">
                            <Zap className="h-4 w-4 mr-2" />
                            Apply Smart Fix
                        </Button>
                    )}
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
};
