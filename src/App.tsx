import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import DataPage from "@/pages/data";
import HyperparametersPage from "./pages/hyperparameters";
import ModelPage from "./pages/model";
import TrainingPage from "./pages/training";

function App() {
    return (
        <div className="w-5/6 mx-auto mt-5">
            <Tabs defaultValue="Data" className="w-full">
                <TabsList className="w-full justify-between">
                    <TabsTrigger value="Data">Data</TabsTrigger>
                    <TabsTrigger value="Hyperparameters">Hyperparameters</TabsTrigger>
                    <TabsTrigger value="Model">Model</TabsTrigger>
                    <TabsTrigger value="Training">Training</TabsTrigger>
                </TabsList>
                <TabsContent value="Data">
                    <DataPage />
                </TabsContent>
                <TabsContent value="Hyperparameters">
                    <HyperparametersPage />
                </TabsContent>
                <TabsContent value="Model">
                    <ModelPage />
                </TabsContent>
                <TabsContent value="Training">
                    <TrainingPage />
                </TabsContent>
            </Tabs>
        </div>
    );
}

export default App;
