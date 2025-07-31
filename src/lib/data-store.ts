import { create } from "zustand";

export type DataPoint = Record<string, unknown>;
export type Dataset = {
	data: DataPoint[];
	columns: string[];
};

export type DataStats = {
	mean: Record<string, number>;
	std: Record<string, number>;
	min: Record<string, number>;
	max: Record<string, number>;
};

export type ChartType = "scatter" | "line" | "histogram" | "pairplot";

export type PresetType =
	| "sinusoid"
	| "spiral"
	| "moons"
	| "iris"
	| "mnist-subset"
	| null;

interface DataState {
	dataset: Dataset | null;
	selectedFeatures: string[];
	selectedTarget: string | null;
	chartType: ChartType;
	xAxis: string | null;
	yAxis: string | null;
	brushEnabled: boolean;
	presetType: PresetType;
	stats: DataStats | null;
	setDataset: (dataset: Dataset) => void;
	setSelectedFeatures: (features: string[]) => void;
	setSelectedTarget: (target: string | null) => void;
	setChartType: (chartType: ChartType) => void;
	setXAxis: (column: string | null) => void;
	setYAxis: (column: string | null) => void;
	toggleBrush: () => void;
	setPresetType: (preset: PresetType) => void;
	loadPreset: (preset: PresetType) => void;
	calculateStats: () => void;
}

// Sample data generators for presets
const generateSinusoid = (): Dataset => {
	const data = Array.from({ length: 100 }, (_, i) => {
		const x = i / 10;
		const y = Math.sin(x) + (Math.random() - 0.5) * 0.5;
		return { x: x, y: y };
	});
	return { data, columns: ["x", "y"] };
};

const generateSpiral = (): Dataset => {
	const data = Array.from({ length: 100 }, (_, i) => {
		const t = i / 10;
		const x = t * Math.cos(t * 2.5);
		const y = t * Math.sin(t * 2.5);
		const class_val = i % 2;
		return { x, y, class: class_val };
	});
	return { data, columns: ["x", "y", "class"] };
};

const generateMoons = (): Dataset => {
	const data = Array.from({ length: 100 }, (_, i) => {
		const half = i >= 50;
		let x, y;
		if (half) {
			x = 1 - Math.cos((i / 50) * Math.PI) + (Math.random() - 0.5) * 0.3;
			y = 0.5 - Math.sin((i / 50) * Math.PI) + (Math.random() - 0.5) * 0.3;
		} else {
			x = Math.cos((i / 50) * Math.PI) + (Math.random() - 0.5) * 0.3;
			y = Math.sin((i / 50) * Math.PI) + (Math.random() - 0.5) * 0.3;
		}
		return { x, y, class: half ? 1 : 0 };
	});
	return { data, columns: ["x", "y", "class"] };
};

// Sample iris-like dataset (simplified)
const generateIris = (): Dataset => {
	const data = [];
	// Generate 3 clusters
	for (let cls = 0; cls < 3; cls++) {
		const meanX = cls * 2;
		const meanY = cls * 1.5;
		const meanWidth = 1 + cls * 0.5;
		const meanHeight = 0.5 + cls * 0.5;

		// Generate 30 samples per cluster
		for (let i = 0; i < 30; i++) {
			const sepal_length = meanX + (Math.random() - 0.5) * 0.5;
			const sepal_width = meanY + (Math.random() - 0.5) * 0.5;
			const petal_length = meanWidth + (Math.random() - 0.5) * 0.3;
			const petal_width = meanHeight + (Math.random() - 0.5) * 0.2;
			data.push({
				sepal_length,
				sepal_width,
				petal_length,
				petal_width,
				species: cls,
			});
		}
	}
	return {
		data,
		columns: [
			"sepal_length",
			"sepal_width",
			"petal_length",
			"petal_width",
			"species",
		],
	};
};

// Simple MNIST subset (simplified representation with 2D features)
const generateMNISTSubset = (): Dataset => {
	const data = Array.from({ length: 100 }, (_, i) => {
		const digit = Math.floor(i / 10); // 0-9 digits
		const feature1 = digit * 0.8 + Math.random() * 0.5;
		const feature2 = digit * 0.5 + Math.random() * 0.5;
		return { feature1, feature2, digit };
	});
	return { data, columns: ["feature1", "feature2", "digit"] };
};

export const useDataStore = create<DataState>((set, get) => ({
	dataset: null,
	selectedFeatures: [],
	selectedTarget: null,
	chartType: "scatter",
	xAxis: null,
	yAxis: null,
	brushEnabled: false,
	presetType: null,
	stats: null,
	setDataset: (dataset) => {
		set({ dataset });
		get().calculateStats();
		if (dataset.columns.length > 0) {
			set({ xAxis: dataset.columns[0] });
			if (dataset.columns.length > 1) {
				set({ yAxis: dataset.columns[1] });
			}
		}
	},
	setSelectedFeatures: (features) => set({ selectedFeatures: features }),
	setSelectedTarget: (target) => set({ selectedTarget: target }),
	setChartType: (chartType) => set({ chartType }),
	setXAxis: (column) => set({ xAxis: column }),
	setYAxis: (column) => set({ yAxis: column }),
	toggleBrush: () => set((state) => ({ brushEnabled: !state.brushEnabled })),
	setPresetType: (preset) => set({ presetType: preset }),
	loadPreset: (preset) => {
		// If preset is null, clear the dataset
		if (preset === null) {
			set({
				dataset: null,
				presetType: null,
				stats: null,
				selectedFeatures: [],
				selectedTarget: null,
				xAxis: null,
				yAxis: null,
			});
			return;
		}

		let dataset: Dataset | null = null;
		switch (preset) {
			case "sinusoid":
				dataset = generateSinusoid();
				break;
			case "spiral":
				dataset = generateSpiral();
				break;
			case "moons":
				dataset = generateMoons();
				break;
			case "iris":
				dataset = generateIris();
				break;
			case "mnist-subset":
				dataset = generateMNISTSubset();
				break;
			default:
				return;
		}
		set({ dataset, presetType: preset });
		get().calculateStats();
		if (dataset.columns.length > 0) {
			set({
				xAxis: dataset.columns[0],
				yAxis: dataset.columns.length > 1 ? dataset.columns[1] : null,
				selectedFeatures: dataset.columns.slice(0, dataset.columns.length - 1),
				selectedTarget: dataset.columns[dataset.columns.length - 1],
			});
		}
	},
	calculateStats: () => {
		const { dataset } = get();
		if (!dataset || dataset.data.length === 0) {
			set({ stats: null });
			return;
		}

		const mean: Record<string, number> = {};
		const std: Record<string, number> = {};
		const min: Record<string, number> = {};
		const max: Record<string, number> = {};

		dataset.columns.forEach((col) => {
			// Skip non-numeric columns
			if (typeof dataset.data[0][col] !== "number") return;

			const values = dataset.data
				.map((d) => d[col])
				.filter((v) => typeof v === "number");

			// Calculate mean
			const sum = values.reduce((acc, val) => acc + val, 0);
			const meanVal = sum / values.length;
			mean[col] = meanVal;

			// Calculate std
			const squaredDiffs = values.map((val) => Math.pow(val - meanVal, 2));
			const avgSquaredDiff =
				squaredDiffs.reduce((acc, val) => acc + val, 0) / values.length;
			std[col] = Math.sqrt(avgSquaredDiff);

			// Calculate min and max
			min[col] = Math.min(...values);
			max[col] = Math.max(...values);
		});

		set({ stats: { mean, std, min, max } });
	},
}));
