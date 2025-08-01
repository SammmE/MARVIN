import type { Layer } from './oscar-store';

export interface ShapeIssue {
  type: 'input_mismatch' | 'layer_mismatch' | 'output_mismatch';
  layerIndex: number;
  layerId: string;
  layerName: string;
  expected: number | number[];
  actual: number | number[];
  message: string;
}

export interface ShapeValidationResult {
  isValid: boolean;
  issues: ShapeIssue[];
  suggestions: ShapeFix[];
}

export interface ShapeFix {
  type: 'adjust_input' | 'adjust_layer' | 'adjust_output' | 'insert_layer';
  layerIndex: number;
  layerId: string;
  description: string;
  action: () => Layer[] | number; // Returns updated layers or new input size
}

export class ShapeValidator {
  static validateModelShape(
    layers: Layer[], 
    inputShape: number | number[], 
    expectedOutputSize?: number
  ): ShapeValidationResult {
    const issues: ShapeIssue[] = [];
    const suggestions: ShapeFix[] = [];

    if (layers.length === 0) {
      return { isValid: true, issues: [], suggestions: [] };
    }

    // Normalize input shape to number for easier processing
    const inputSize = Array.isArray(inputShape) ? inputShape[inputShape.length - 1] : inputShape;
    let currentSize = inputSize;

    // Check each layer's input/output compatibility
    for (let i = 0; i < layers.length; i++) {
      const layer = layers[i];

      // For Dense layers, check if input size matches
      if (layer.type === 'Dense') {
        // First layer: check if it matches the input size
        if (i === 0) {
          // For first layer, we can be flexible - TensorFlow will handle reshaping
          // But we should warn if there's a dramatic size mismatch
          if (layer.units && inputSize > layer.units * 10) {
            issues.push({
              type: 'input_mismatch',
              layerIndex: i,
              layerId: layer.id,
              layerName: `Layer ${i + 1} (${layer.type})`,
              expected: inputSize,
              actual: layer.units || 0,
              message: `First layer has ${layer.units} units but input has ${inputSize} features. This may cause information loss.`
            });

            suggestions.push({
              type: 'adjust_layer',
              layerIndex: i,
              layerId: layer.id,
              description: `Increase first layer to ${Math.max(layer.units || 0, Math.ceil(inputSize / 2))} units to better handle input features`,
              action: () => {
                const newLayers = [...layers];
                newLayers[i] = { ...layer, units: Math.max(layer.units || 0, Math.ceil(inputSize / 2)) };
                return newLayers;
              }
            });
          }
        }

        // Update current size for next layer
        currentSize = layer.units || currentSize;
      }
      
      // Add more layer type checks here as needed
      // Example for Convolutional layers:
      // else if (layer.type === 'Conv2D') { ... }
    }

    // Check output layer compatibility
    if (expectedOutputSize && layers.length > 0) {
      const lastLayer = layers[layers.length - 1];
      const lastLayerOutputSize = lastLayer.units || currentSize;

      if (lastLayerOutputSize !== expectedOutputSize) {
        issues.push({
          type: 'output_mismatch',
          layerIndex: layers.length - 1,
          layerId: lastLayer.id,
          layerName: `Output Layer (${lastLayer.type})`,
          expected: expectedOutputSize,
          actual: lastLayerOutputSize,
          message: `Output layer has ${lastLayerOutputSize} units but expected ${expectedOutputSize} for the target data.`
        });

        suggestions.push({
          type: 'adjust_output',
          layerIndex: layers.length - 1,
          layerId: lastLayer.id,
          description: `Adjust output layer to ${expectedOutputSize} units to match target data`,
          action: () => {
            const newLayers = [...layers];
            newLayers[layers.length - 1] = { ...lastLayer, units: expectedOutputSize };
            return newLayers;
          }
        });
      }
    }

    // Check for layers that are too small (potential bottlenecks)
    for (let i = 1; i < layers.length - 1; i++) { // Skip first and last layers
      const layer = layers[i];
      const prevLayer = layers[i - 1];
      
      if (layer.type === 'Dense' && prevLayer.type === 'Dense') {
        const currentUnits = layer.units || 0;
        const prevUnits = prevLayer.units || 0;
        
        // If current layer is less than 10% of previous layer, it might be too small
        if (currentUnits < prevUnits * 0.1 && currentUnits < 16) {
          issues.push({
            type: 'layer_mismatch',
            layerIndex: i,
            layerId: layer.id,
            layerName: `Layer ${i + 1} (${layer.type})`,
            expected: Math.ceil(prevUnits * 0.25),
            actual: currentUnits,
            message: `Layer ${i + 1} has only ${currentUnits} units, which may create a bottleneck after ${prevUnits} units.`
          });

          suggestions.push({
            type: 'adjust_layer',
            layerIndex: i,
            layerId: layer.id,
            description: `Increase layer to ${Math.ceil(prevUnits * 0.25)} units to avoid bottleneck`,
            action: () => {
              const newLayers = [...layers];
              newLayers[i] = { ...layer, units: Math.ceil(prevUnits * 0.25) };
              return newLayers;
            }
          });
        }
      }
    }

    return {
      isValid: issues.length === 0,
      issues,
      suggestions
    };
  }

  static getInputShapeFromData(dataset: any): number {
    if (!dataset || !dataset.data || dataset.data.length === 0) {
      return 784; // Default fallback
    }

    const firstRow = dataset.data[0];
    const featureColumns = dataset.featureColumns || [];
    
    if (featureColumns.length > 0) {
      return featureColumns.length;
    }

    // Count numeric columns (excluding target)
    const targetColumn = dataset.targetColumn || 'y';
    let featureCount = 0;
    
    for (const key in firstRow) {
      if (key !== targetColumn && typeof firstRow[key] === 'number') {
        featureCount++;
      }
    }

    return featureCount || 1;
  }

  static getExpectedOutputSize(dataset: any, problemType: 'regression' | 'classification'): number {
    if (problemType === 'regression') {
      return 1; // Single continuous output
    }

    if (!dataset || !dataset.data || dataset.data.length === 0) {
      return 2; // Default binary classification
    }

    const targetColumn = dataset.targetColumn || 'y';
    const uniqueValues = new Set();
    
    for (const row of dataset.data) {
      if (row[targetColumn] !== undefined) {
        uniqueValues.add(row[targetColumn]);
      }
    }

    // For binary classification, we can use 1 output with sigmoid
    // For multi-class, we need one output per class
    return uniqueValues.size === 2 ? 1 : uniqueValues.size;
  }

  static applySuggestionFixes(suggestions: ShapeFix[]): { layers?: Layer[], inputShape?: number } {
    const result: { layers?: Layer[], inputShape?: number } = {};
    
    for (const suggestion of suggestions) {
      const actionResult = suggestion.action();
      
      if (Array.isArray(actionResult)) {
        result.layers = actionResult;
      } else if (typeof actionResult === 'number') {
        result.inputShape = actionResult;
      }
    }
    
    return result;
  }
}
