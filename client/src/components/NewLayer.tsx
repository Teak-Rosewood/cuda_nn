import { Layer } from "../Managers/ModelManger";

interface NewLayerProp {
    layers: Layer[];
    setLayers: (layers: Layer[]) => void;
    setEpochs: (epochs: number) => void;
}
let LAYER_ID = 1;

const NewLayer = ({ layers, setLayers, setEpochs }: NewLayerProp) => {
    const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        const newLayer: Layer = {
            layerId: LAYER_ID++,
            units: (e.target as unknown as HTMLInputElement[])[0].valueAsNumber,
            activation: (e.target as unknown as HTMLInputElement[])[1].value,
        };
        setLayers([...layers, newLayer]);
    };
    return (
        <>
            <form className="flex flex-col" onSubmit={handleSubmit}>
                <label className="p-1">
                    Units:
                    <input
                        required={true}
                        className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-1 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
                        type="number"
                        name="units"
                    />
                </label>
                <label className="p-1">
                    Activation:
                    <select name="activation">
                        <option value="ReLu">ReLu</option>
                        <option value="None">None</option>
                    </select>
                </label>
                <button
                    className="py-2.5 px-5 me-2 mb-2 text-sm font-medium text-gray-900 focus:outline-none bg-white rounded-lg border border-gray-200 hover:bg-gray-100 hover:text-blue-700 focus:z-10 focus:ring-4 focus:ring-gray-100 dark:focus:ring-gray-700 dark:bg-gray-800 dark:text-gray-400 dark:border-gray-600 dark:hover:text-white dark:hover:bg-gray-700"
                    type="submit"
                >
                    Add Layer
                </button>
                <label className="p-1">
                Epochs:
                <input
                    className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-1 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
                    type="number"
                    name="epochs"
                    onChange={(e) => setEpochs(e.target.valueAsNumber)}
                />
            </label>
            </form>
            
        </>
    );
};

export default NewLayer;
