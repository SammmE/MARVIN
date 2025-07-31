import { useEffect, useState } from "react";
import {
	LineChart,
	Line,
	XAxis,
	YAxis,
	CartesianGrid,
	Tooltip,
	ResponsiveContainer,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";

interface ActivationFunctionGraphProps {
	functionCode: string;
}

export function ActivationFunctionGraph({
	functionCode,
}: ActivationFunctionGraphProps) {
	const [data, setData] = useState<Array<{ x: number; y: number }>>([]);
	const [error, setError] = useState<string | null>(null);

	useEffect(() => {
		try {
			// Generate sample points from -5 to 5
			const points = [];

			// Create a safe function wrapper with limited context
			const createActivationFunction = new Function(
				"code",
				`
        try {
          // Extract just the function body if it's a full function declaration
          const funcMatch = code.match(/function\\s+\\w*\\s*\\(\\w+\\)\\s*{([\\s\\S]*)}/);
          const arrowFuncMatch = code.match(/\\(\\w+\\)\\s*=>\\s*{([\\s\\S]*)}/);
          const simpleTernaryMatch = code.match(/(\\w+)\\s*=>\\s*([^{][\\s\\S]*)/);
          
          let funcBody;
          let paramName = 'x';
          
          if (funcMatch) {
            funcBody = funcMatch[1];
            paramName = code.match(/function\\s+\\w*\\s*\\((\\w+)\\)/)[1] || 'x';
          } else if (arrowFuncMatch) {
            funcBody = arrowFuncMatch[1];
            paramName = code.match(/\\((\\w+)\\)\\s*=>/)[1] || 'x';
          } else if (simpleTernaryMatch) {
            funcBody = \`return \${simpleTernaryMatch[2]}\`;
            paramName = simpleTernaryMatch[1];
          } else {
            // Just try to use the code as is
            funcBody = code.includes('return') ? code : \`return \${code}\`;
          }
          
          // Create the function
          return new Function(paramName, funcBody);
        } catch (e) {
          throw new Error('Failed to parse function: ' + e.message);
        }
      `,
			);

			// Try to create the activation function
			const activationFunc = createActivationFunction(functionCode);

			// Generate data points
			for (let x = -5; x <= 5; x += 0.2) {
				try {
					const y = activationFunc(x);
					// Check if y is a valid number
					if (typeof y === "number" && !isNaN(y) && isFinite(y)) {
						points.push({ x, y });
					}
				} catch (e) {
					console.error("Error calculating point at x=", x, e);
				}
			}

			setData(points);
			setError(null);
		} catch (e) {
			console.error("Error rendering activation function:", e);
			setError(
				e instanceof Error
					? e.message
					: "Failed to parse or execute the function",
			);
			setData([]);
		}
	}, [functionCode]);

	if (error) {
		return (
			<Card className="mt-4">
				<CardHeader>
					<CardTitle>Activation Function Preview</CardTitle>
				</CardHeader>
				<CardContent>
					<div className="text-red-500 p-4 bg-red-50 rounded-md">
						Error: {error}
					</div>
				</CardContent>
			</Card>
		);
	}

	return (
		<Card className="mt-4">
			<CardHeader>
				<CardTitle>Activation Function Preview</CardTitle>
			</CardHeader>
			<CardContent>
				<div className="h-[200px]">
					<ResponsiveContainer width="100%" height="100%">
						<LineChart
							data={data}
							margin={{
								top: 5,
								right: 30,
								left: 20,
								bottom: 5,
							}}
						>
							<CartesianGrid strokeDasharray="3 3" />
							<XAxis
								dataKey="x"
								domain={[-5, 5]}
								label={{ value: "Input", position: "bottom" }}
							/>
							<YAxis
								domain={[-2, 2]}
								label={{ value: "Output", angle: -90, position: "left" }}
							/>
							<Tooltip
								formatter={(value) => parseFloat(value.toString()).toFixed(4)}
							/>
							<Line
								type="monotone"
								dataKey="y"
								stroke="#8884d8"
								strokeWidth={2}
								dot={false}
								name="f(x)"
							/>
						</LineChart>
					</ResponsiveContainer>
				</div>
			</CardContent>
		</Card>
	);
}
