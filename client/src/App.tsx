import { useEffect, useState } from "react";
import "./App.css";
import { ModelManager, Layer } from "./Managers/ModelManger";
import NewLayer from "./components/NewLayer";
import LayerList from "./components/LayerList";
import FeatureSelector from "./components/FeatureSelector";

function App() {
    const [layers, setLayers] = useState<Layer[]>([]);
    const [status, setStatus] = useState<string>("Dataset Upload");
    const [features, setFeatures] = useState<string[]>([]);
    const [epochs, setEpochs] = useState<number>(0);
    const [file, setFile] = useState<File | null>(null);
    const [modelFile, setModelFile] = useState<File | null>(null);
    const [model, setModel] = useState<ModelManager>();

    useEffect(() => {
        const init_model = new ModelManager();
        setModel(init_model);
    }, []);

    const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file && file.type !== "text/csv") {
            setStatus("Please upload a .csv file");
            event.target.value = "";
        } else if (file) {
            setFile(file);
        }
    };

    const uploadDataset = async () => {
        if (file) {
            setStatus("Feature Selection");
            const new_feature = await model?.uploadDataset(file);
            setFeatures(new_feature);
        } else {
            setStatus("No file selected");
        }
    };

    const compileModel = async () => {
        setStatus("Model is Compiling...");
        const res = await model?.compileModel(layers);
        setStatus(res);
    };

    const fitModel = async () => {
        setStatus("Model is Training...");
        if (epochs === 0) {
            setStatus("Epochs has to be greater than 0");
            return;
        }
        const res = await model?.fitModel(epochs, layers);
        // setModelFile(res.model);
        setStatus(res.message);
    };

    const downloadModel = () => {
        if (modelFile) {
            const url = URL.createObjectURL(modelFile);
            const a = document.createElement("a");
            a.href = url;
            a.download = "model.csv";
            document.body.appendChild(a);
            a.click();
            a.remove();
        } else {
            setStatus("Model has not been trained yet");
        }
    };
    return (
        <>
            <div className="p-4">
                <h1>Artifical Neural Network Builder</h1>
                <div className="text-green-500"> Status: {status}</div>
            </div>

            <div className="p-4">
                <h2>Upload the Dataset</h2>
                <input
                    className="block w-full text-sm text-gray-900 border border-gray-300 rounded-lg cursor-pointer bg-gray-50 dark:text-gray-400 focus:outline-none dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400"
                    type="file"
                    onChange={handleFileUpload}
                />
                <button
                    className="py-2.5 px-5 me-2 mb-2 text-sm font-medium text-gray-900 focus:outline-none bg-white rounded-lg border border-gray-200 hover:bg-gray-100 hover:text-blue-700 focus:z-10 focus:ring-4 focus:ring-gray-100 dark:focus:ring-gray-700 dark:bg-gray-800 dark:text-gray-400 dark:border-gray-600 dark:hover:text-white dark:hover:bg-gray-700"
                    onClick={uploadDataset}
                >
                    Upload Dataset
                </button>
            </div>

            <FeatureSelector object={model} setStatus={setStatus} features={features} />

            <div className="p-4">
                <h2>Build Your Model Structure</h2>
                <div className="grid grid-cols-3 items-center">
                    <NewLayer layers={layers} setLayers={setLayers} setEpochs={setEpochs} />
                    <LayerList layers={layers} setLayers={setLayers} />
                </div>
                <button
                    className="py-2.5 px-5 me-2 mb-2 text-sm font-medium text-gray-900 focus:outline-none bg-white rounded-lg border border-gray-200 hover:bg-gray-100 hover:text-blue-700 focus:z-10 focus:ring-4 focus:ring-gray-100 dark:focus:ring-gray-700 dark:bg-gray-800 dark:text-gray-400 dark:border-gray-600 dark:hover:text-white dark:hover:bg-gray-700"
                    onClick={compileModel}
                >
                    Compile Model
                </button>
                <button
                    className="py-2.5 px-5 me-2 mb-2 text-sm font-medium text-gray-900 focus:outline-none bg-white rounded-lg border border-gray-200 hover:bg-gray-100 hover:text-blue-700 focus:z-10 focus:ring-4 focus:ring-gray-100 dark:focus:ring-gray-700 dark:bg-gray-800 dark:text-gray-400 dark:border-gray-600 dark:hover:text-white dark:hover:bg-gray-700"
                    onClick={fitModel}
                >
                    Fit Model
                </button>
                <button
                    className="py-2.5 px-5 me-2 mb-2 text-sm font-medium text-gray-900 focus:outline-none bg-white rounded-lg border border-gray-200 hover:bg-gray-100 hover:text-blue-700 focus:z-10 focus:ring-4 focus:ring-gray-100 dark:focus:ring-gray-700 dark:bg-gray-800 dark:text-gray-400 dark:border-gray-600 dark:hover:text-white dark:hover:bg-gray-700"
                    onClick={downloadModel}
                >
                    Download Model
                </button>
            </div>
        </>
    );
}

export default App;
