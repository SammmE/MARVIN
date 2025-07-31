import { DataControls } from "@/components/data-controls";
import { DataPreview } from "@/components/data-preview";
import { DataVisualization } from "@/components/data-visualization";

export default function DataPage() {
	return (
		<div className="space-y-8 py-4">
			<div className="grid grid-cols-1 md:grid-cols-3 gap-8">
				<div className="md:col-span-1">
					<DataControls />
				</div>
				<div className="md:col-span-2 space-y-8">
					<DataVisualization />
					<DataPreview />
				</div>
			</div>
		</div>
	);
}
