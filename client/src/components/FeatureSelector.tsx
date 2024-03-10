import { ModelManager } from "../Managers/ModelManger";

interface FeatureSelectorProps {
    object: ModelManager | undefined;
    setStatus: (status: string) => void;
    features: string[];
}

const FeatureSelector = ({ object, setStatus, features }: FeatureSelectorProps) => {
   
    const setPredictionFeature = async (feature: string) => {
        const status = await object?.setPredictionFeature(feature);
        if (!status) {
            setStatus("Failed to set the prediction feature");
        } else setStatus("Compile the Model");
    };

    if (features.length === 0)
        return (
            <>
                <h2>Choose the value you want to predict</h2>
                <div className="text-red-500">Dataset Not Uploaded or Features dont exist</div>
            </>
        );

    return (
        <div className="p-4">
            <h2>Choose the value you want to predict</h2>
            {features.map((feature, index) => {
                return (
                        <button
                            onClick={() => setPredictionFeature(feature)}
                            className="p-1 text-white bg-gray-800 hover:bg-gray-900 focus:outline-none focus:ring-4 focus:ring-gray-300 font-medium rounded-lg text-sm px-5 py-2.5 me-2 mb-2 dark:bg-gray-800 dark:hover:bg-gray-700 dark:focus:ring-gray-700 dark:border-gray-700"
                            key={index + "-button"}
                        >
                            {feature}
                        </button>
                );
            })}
        </div>
    );
};

export default FeatureSelector;
