import { Layer } from "../Managers/ModelManger";

interface LayerListProp {
    layers: Layer[];
    setLayers: (layers: Layer[]) => void;
}
const LayerList = ({ layers, setLayers }: LayerListProp) => {
    const deleteLayer = (id: number) => {
        const newLayers = layers.filter((layer) => layer.layerId !== id);
        setLayers(newLayers);
        if(newLayers.length === 0) setLayers([]);
    };

    return (
        <>
            <div className="col-span-2 align-middle" key="LayerList">
                <div className="flex justify-between">
                    <div className="p-1"> Layer ID</div>
                    <div className="p-1"> Units</div>
                    <div className="p-1"> Activation</div>
                    <div className="p-1"> Delete Layer</div>
                </div>
                {layers.map((layer, index) => {
                    return (
                        <div className="flex justify-between" key={index}>
                            <div className="p-1 align-middle" key={index + "-id"}>
                                {layer.layerId}
                            </div>
                            <div className="p-1" key={index + "-units"}>
                                {layer.units}
                            </div>
                            <div className="p-1" key={index + "-activation"}>
                                {layer.activation}
                            </div>
                            <button
                                onClick={() => deleteLayer(layer.layerId)}
                                className="p-1 text-white bg-gray-800 hover:bg-gray-900 focus:outline-none focus:ring-4 focus:ring-gray-300 font-medium rounded-lg text-sm px-5 py-2.5 me-2 mb-2 dark:bg-gray-800 dark:hover:bg-gray-700 dark:focus:ring-gray-700 dark:border-gray-700"
                                key={index + "-button"}
                            >
                                Delete
                            </button>
                        </div>
                    );
                })}
            </div>
        </>
    );
};

export default LayerList;
